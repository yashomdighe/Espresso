import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from models.attention import MultiHeadedAttention
from models.gauss_render import Renderer
from models.gaussian_model import GaussianModel
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from loss.loss_utils import l1_loss, ssim

class Espresso(L.LightningModule):
    def __init__(self, tokenizer, text_encoder, pcd_encoder, d_model):
        super(Espresso, self).__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.pcd_encoder = pcd_encoder

        self.d_model = d_model
        self.text_proj = nn.Linear(768, self.d_model)
        self.mha1 = MultiHeadedAttention(h=8, d_model=self.d_model, dropout=0.1)

        self.fc1 = nn.Linear(self.d_model, 2048*3)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(2048, 8192)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(8192, 8192)
        self.drop3 = nn.Dropout(0.4)
        self.fc4 = nn.Linear(8192, 131072)
        self.drop4 = nn.Dropout(0.4)
        # self.fc5 = nn.Linear(131072, 131072)
        # self.drop5 = nn.Dropout(0.4)
        # self.fc4 = nn.Linear(32768, 150000)
        # self.drop4 = nn.Dropout(0.4)
        self.im_tf = ToPILImage()
        # self.fc5 = nn.Linear(131072, 150000)
        # self.drop5 = nn.Dropout(0.4)

        self.loss = nn.MSELoss()


    def render(
            self,
            means3D, mask, opacity, scales, rotations, shs, active_sh_degree, 
            bg_color : torch.Tensor,
            FovX, FovY, 
            world_view_transform,
            full_proj_transform,
            camera_center,
            image_height = 128,
            image_width = 128, 
            scaling_modifier = 1.0, 
            separate_sh = False, 
            override_color = None, 
            use_trained_exp=False
        ):
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(FovX * 0.5)
        tanfovy = math.tan(FovY * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(image_height),
            image_width=int(image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=active_sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points
        cov3D_precomp = None
        colors_precomp = None

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        return rendered_image
    
    def forward(self, means3D, mask, opacity, scales, rotations, shs, active_sh_degree, task_desc, FovX, FovY, world_view_transform, full_proj_transform, camera_center, inference=False):
        # xyz = G.get_xyz
        # xyz, mask = self.pad_input(xyz, target_size=150000)
        # G._xyz = xyz

        # print(task_desc, type(task_desc))
        tokens = self.tokenizer(task_desc, return_tensors='pt', truncation=True, max_length=128)
        # tokens = {key: value.to(self.device) for key, value in tokens.items()}
        text_emb = self.text_proj(self.text_encoder(**tokens.to("cuda")).text_embeds)
        # xyz = xyz[None,:,:]
        pcd_emb = self.pcd_encoder(means3D.transpose(2, 1))

        x = F.relu(self.mha1(text_emb, pcd_emb, pcd_emb))
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x.reshape(3,2048))))
        x = F.relu(self.drop3(self.fc3(x)))
        # x = self.drop4(self.fc4(x))
        # x = self.drop4(self.fc4(x))
        # x = torch.mul(mask, self.drop4(self.fc4(x)))
        x = torch.mul(mask, 2*F.tanh(self.drop4(self.fc4(x))))

        # G._xyz = G._xyz + x
        means3D += x.reshape(131072,3)
        # print(means3D)
        # exit(1)

        if inference:
            return means3D

        rendered_img = self.render(
            means3D[0], mask[0], opacity[0], scales[0], rotations[0], shs[0], active_sh_degree[0],
            torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device),
            FovX,
            FovY,
            world_view_transform,
            full_proj_transform,
            camera_center,
        )

        return rendered_img

    def training_step(self, batch, batch_idx):

        means3D, mask, opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, task_desc, target = batch
        # print(task_desc, type(task_desc))
        output = self(means3D, mask, opacity, scales, rotations, shs, active_sh_degree, task_desc, FovX, FovY, world_view_transform, full_proj_transform, camera_center)
        loss = self.compute_loss(output, target[0])
        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        means3D, mask, opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, task_desc, target = batch
        output = self(means3D, mask, opacity, scales, rotations, shs, active_sh_degree, task_desc, FovX, FovY, world_view_transform, full_proj_transform, camera_center)
        loss = self.compute_loss(output, target[0])
        self.log("val_loss", loss.item())
        return loss
    
    def inference_step(self, G, task_desc, FovX, FovY, world_view_transform, full_proj_transform, camera_center):
        with torch.no_grad():
            G_pred = self(G, task_desc, FovX, FovY, world_view_transform, full_proj_transform, camera_center, inference=True)
        return G_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def compute_loss(self, output, target):
        """Compute loss for training and validation."""
        # l1 = l1_loss(output, target)
        # ssim_loss = ssim(output, target)
        # lambda_dssim = 0.2
        op = self.im_tf(output)
        op.save("render.png")

        gt = self.im_tf(target)
        gt.save("gt.png")
        
        # loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_loss)
        return self.loss(output, target).sum()






        


