import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import os
from PIL import Image
import torchvision
from torchvision.transforms import ToTensor, ToPILImage
from models.gauss_render_multiview import Renderer
from models.gaussian_model import GaussianModel
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from pytorch3d.loss import chamfer_distance
from loss.loss_utils import l1_loss, ssim, rigidity_loss2, local_rigid_loss
from scipy.spatial import cKDTree
from torch_cluster import radius_graph, knn_graph

class EspressoV5(L.LightningModule):
    def __init__(self, out_channels, version):
        super(EspressoV5, self).__init__()
        self.version = version
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

    def forward(self, means3D, pt_mask, opacity, scales, rotations, shs, active_sh_degree, FovX, FovY, world_view_transform, full_proj_transform, camera_center, inference=False):
        
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
        # x = l0_xyz + 0.5*F.tanh(self.conv2(x))
        x = l0_xyz + self.conv2(x)
        # # x = F.log_softmax(x, dim=1)

        x = x.permute(0, 2, 1) * pt_mask.unsqueeze(-1)
        # print(x.size(), pt_mask.size())

        # # print(x.size())
        means3D = means3D + x
        # means3D = x
        # # print(means3D)
        # exit(1)

        if inference:
            return x

        rendered_img = self.renderer(
            means3D[0], opacity[0], scales[0], rotations[0], shs[0], active_sh_degree[0],
            torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device),
            FovX,
            FovY,
            world_view_transform,
            full_proj_transform,
            camera_center,
        )

        # exit(1)
        return rendered_img, means3D, x

    def training_step(self, batch, batch_idx):

        means3D,  opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, task_desc, target_im, target_means, pt_mask ,im_mask = batch
        # print(task_desc, type(task_desc))
        output_im, output_means, deltas = self(means3D, pt_mask, opacity, scales, rotations, shs, active_sh_degree, FovX, FovY, world_view_transform, full_proj_transform, camera_center)
        
        # loss_2d, loss_3d, loss_rigid = self.compute_loss(output_im, target_im[0], output_means, target_means, means3D, mask, deltas, batch_idx, "train")
        # loss = loss_2d + loss_3d + loss_rigid
        loss_2d_im, loss_2d_masked, loss_rigid = self.compute_loss(output_im, target_im[0], output_means, target_means, means3D, im_mask, pt_mask, deltas, batch_idx, "train")
        loss = loss_2d_im + 3.0*loss_2d_masked + 4.0*loss_rigid

        self.log("train_loss", loss.item(), batch_size=1)
        self.log("train_2d_loss", loss_2d_im.item(), batch_size=1)
        # self.log("train_2d_loss", loss_2d_masked.item(), batch_size=1)
        self.log("train_2d_mask_loss", loss_2d_masked.item(), batch_size=1)
        # self.log("train_3d_loss", loss_3d.item(), batch_size=1) 
        self.log("train_rigid_loss", loss_rigid.item(), batch_size=1) 

        return loss

    def validation_step(self, batch, batch_idx):
        means3D,  opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, task_desc, target_im, target_means, pt_mask ,im_mask = batch
        output_im, output_means, deltas = self(means3D, pt_mask, opacity, scales, rotations, shs, active_sh_degree, FovX, FovY, world_view_transform, full_proj_transform, camera_center)

        # loss_2d, loss_3d, loss_rigid = self.compute_loss(output_im, target_im[0], output_means, target_means, means3D, mask, deltas, batch_idx, "val")
        # loss = loss_2d + loss_3d + loss_rigid
        loss_2d_im, loss_2d_masked, loss_rigid = self.compute_loss(output_im, target_im[0], output_means, target_means, means3D, im_mask, pt_mask, deltas, batch_idx, "val")
        loss = loss_2d_im + 3.0*loss_2d_masked + 4.0*loss_rigid
        # loss = (1.0 - self.lambda_loss) * loss_2d + self.lambda_loss * (loss_3d)
        self.log("val_loss", loss.item(), batch_size=1)
        self.log("val_2d_loss", loss_2d_im.item(), batch_size=1)
        # self.log("val_2d_loss", loss_2d_masked.item(), batch_size=1)
        self.log("val_2d_mask_loss", loss_2d_masked.item(), batch_size=1)
        # self.log("val_3d_loss", loss_3d.item(), batch_size=1)
        self.log("train_rigid_loss", loss_rigid.item(), batch_size=1)
        
        return loss
    
    def inference_step(self, means3D,  opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, task_desc, target_im, target_means):
        with torch.no_grad():
            pred_means = self(means3D, opacity, scales, rotations, shs, active_sh_degree, FovX, FovY, world_view_transform, full_proj_transform, camera_center, inference=True)
        return pred_means

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def compute_loss(self, output_im, target_im, output_means, target_means, input_means, mask, pt_mask, deltas, batch_idx, type):
        """Compute loss for training and validation."""

        V, C, H, W = output_im.shape  # B=batch size, V=6 views

        # print(output_im.shape)
        # exit(1)
        # output_flat = output_im.view(B * V, C, H, W)
        # target_flat = target_im.view(B * V, C, H, W)
        # mask_flat = mask.view(B * V, 1, H, W).float()
        l1 = l1_loss(output_im, target_im)
        ssim_loss = ssim(output_im, target_im)
        # loss_2d = ssim(output_im, target_im)
        lambda_dssim = 0.2
        loss_2d_im = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_loss)

        #######################
        # Masked MSE
        #######################
        mask = mask.float()

        # # Compute the squared error
        squared_error = (target_im - output_im) ** 2

        # # Use torch.where to filter only masked pixels
        masked_squared_error = torch.where(mask == 1, squared_error, torch.tensor(0.0))

        # # Compute the mean of the non-zero elements
        loss_2d_masked = masked_squared_error.sum() / mask.sum()
        # loss_2d = 0.8 * loss_2d_im + 0.2 * loss_2d_masked

        #######################
        # Rigidity Regularization
        #######################
        masked_ind = torch.where(pt_mask.squeeze(0)==1)
        src, dst = radius_graph(input_means.detach().squeeze(0)[masked_ind], r=1e-2)
        # src, dst = knn_graph(input_means.detach().squeeze(0)[masked_ind], k=20, loop=True)
        # print(input_means.shape)
        # print(torch.unique(src).shape)
        # print(src.shape)
        # print(input_means.squeeze(0).shape)
        prev_diff = input_means.squeeze(0)[src] - input_means.squeeze(0)[dst]
        # print(prev_diff.shape)
        curr_diff = output_means.squeeze(0)[src] - output_means.squeeze(0)[dst]
        # print(curr_diff.shape)

        loss_rigid = torch.mean(torch.sqrt(torch.sum((prev_diff - curr_diff) ** 2, dim=1)+ 1e-20))
        # print(loss_rigid)
        # print(((prev_diff - curr_diff) ** 2).shape)
        # exit(1)
        # print(diff.shape)
        # exit(1)

        cmp = torch.cat((target_im, output_im), dim=2)
        if not os.path.exists(f"output/{self.version}/renders/{type}/{self.current_epoch}"):
            os.makedirs(f"output/{self.version}/renders/{type}/{self.current_epoch}")
        # torchvision.utils.save_image(cmp, "comparison.png")
        torchvision.utils.save_image(cmp, f"output/{self.version}/renders/{type}/{self.current_epoch}/comparison{batch_idx}.png")

        # return loss_2d, loss_rigid
        return loss_2d_im, loss_2d_masked, loss_rigid
        # return loss_2d, loss_3d, loss_rigid
        # return self.loss(output, target).sum()






        

