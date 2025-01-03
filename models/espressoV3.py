import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
import os
from PIL import Image
import torchvision
from torchvision.transforms import ToTensor, ToPILImage
from models.attention import MultiHeadedAttention
from models.gauss_render import Renderer
from models.gaussian_model import GaussianModel
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from pytorch3d.loss import chamfer_distance
from loss.loss_utils import l1_loss, ssim

class EspressoV3(L.LightningModule):
    def __init__(self, out_channels):
        super(EspressoV3, self).__init__()
        self.lambda_loss = 0.8
        # self.name = name
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, out_channels, 1)

        self.im_tf = ToPILImage()
        self.renderer = Renderer()

        self.im_loss = nn.MSELoss()

    def forward(self, means3D, opacity, scales, rotations, shs, active_sh_degree, FovX, FovY, world_view_transform, full_proj_transform, camera_center, inference=False):
        
        xyz = means3D.transpose(2, 1)
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        # print(l0_xyz.size())
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))

        # print(x.size())
        x = l0_xyz + 3*F.tanh(self.conv2(x))
        # # x = F.log_softmax(x, dim=1)

        x = x.permute(0, 2, 1)
        # # print(x.size())
        # # means3D = means3D + x
        # means3D = x
        # # print(means3D)
        # # exit(1)

        if inference:
            return x

        rendered_img = self.renderer(
            x[0], opacity[0], scales[0], rotations[0], shs[0], active_sh_degree[0],
            torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device),
            FovX,
            FovY,
            world_view_transform,
            full_proj_transform,
            camera_center,
        )

        # exit(1)
        return rendered_img, x

    def training_step(self, batch, batch_idx):

        means3D,  opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, task_desc, target_im, target_means = batch
        # print(task_desc, type(task_desc))
        output_im, output_means = self(means3D, opacity, scales, rotations, shs, active_sh_degree, FovX, FovY, world_view_transform, full_proj_transform, camera_center)
        loss_2d, loss_3d = self.compute_loss(output_im, target_im[0], output_means, target_means, batch_idx, "train")

        loss = (1.0 - self.lambda_loss) * loss_2d +  self.lambda_loss * (loss_3d)
        self.log("train_loss", loss.item(), batch_size=1)
        self.log("train_2d_loss", loss_2d.item(), batch_size=1)
        self.log("train_3d_loss", loss_3d.item(), batch_size=1) 

        return loss

    def validation_step(self, batch, batch_idx):
        means3D,  opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, task_desc, target_im, target_means = batch
        output_im, output_means = self(means3D, opacity, scales, rotations, shs, active_sh_degree, FovX, FovY, world_view_transform, full_proj_transform, camera_center)

        loss_2d, loss_3d = self.compute_loss(output_im, target_im[0], output_means, target_means, batch_idx, "val")
        loss = (1.0 - self.lambda_loss) * loss_2d + self.lambda_loss * (loss_3d)
        self.log("val_loss", loss.item(), batch_size=1)
        self.log("val_2d_loss", loss_2d.item(), batch_size=1)
        self.log("val_3d_loss", loss_3d.item(), batch_size=1)
        
        return loss
    
    def inference_step(self, means3D,  opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, task_desc, target_im, target_means):
        with torch.no_grad():
            pred_means = self(means3D, opacity, scales, rotations, shs, active_sh_degree, FovX, FovY, world_view_transform, full_proj_transform, camera_center, inference=True)
        return pred_means

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def compute_loss(self, output_im, target_im, output_means, target_means, batch_idx, type):
        """Compute loss for training and validation."""
        l1 = l1_loss(output_im, target_im)
        ssim_loss = ssim(output_im, target_im)
        # loss_2d = ssim(output_im, target_im)
        lambda_dssim = 0.2

        # print(target.device)
        # print(output_im.device)
        loss_2d = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_loss)
        # weight_l3d = 0.3
        # loss_2d = self.im_loss(output_im, target_im).sum()
        # loss_3d, loss_3d_normals = chamfer_distance(output_means, target_means, point_reduction="sum")
        loss_3d, loss_3d_normals = chamfer_distance(output_means, target_means)
        # loss = loss_2d + weight_l3d*loss_3d


        cmp = torch.cat((target_im, output_im), dim=2)
        if not os.path.exists(f"renders4/{type}/{self.current_epoch}"):
            os.makedirs(f"renders4/{type}/{self.current_epoch}")
        # cmp.save(f"renders/{type}/{self.current_epoch}/comparison{batch_idx}.png")
        # cmp.save(f"comparison.png")
        # torchvision.utils.save_image(cmp, "comparison.png")
        torchvision.utils.save_image(cmp, f"renders4/{type}/{self.current_epoch}/comparison{batch_idx}.png")

        return loss_2d, loss_3d
        # return self.loss(output, target).sum()






        


