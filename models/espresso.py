import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
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
        self.fc3 = nn.Linear(8192, 32768)
        self.drop3 = nn.Dropout(0.4)
        self.fc4 = nn.Linear(32768, 150000)
        self.drop4 = nn.Dropout(0.4)
        # self.fc5 = nn.Linear(131072, 150000)
        # self.drop5 = nn.Dropout(0.4)

    def render( 
            self,
            means3D, opacity, scales, rotations, shs, active_sh_degree,
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
        """
        Render the scene. 
        
        Background tensor (bg_color) must be on GPU!
        """
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
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
            sh_degree= active_sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # means3D = pc.get_xyz
        means2D = screenspace_points
        # opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        cov3D_precomp = None

        # scales = pc.get_scaling
        # rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        # shs = None
        colors_precomp = None
        # if override_color is None:  
                # shs = shs
        # else:
            # colors_precomp = override_color

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

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        # return {"render": rendered_image,
        #         "viewspace_points": screenspace_points,
        #         "visibility_filter" : radii > 0,
        #         "radii": radii}
        return rendered_image
    
    def forward(self, means3D, mask, opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, task_desc, inference=False):
        # xyz = G.get_xyz
        # xyz, mask = self.pad_input(xyz, target_size=150000)
        # G._xyz = xyz

        # print(task_desc, type(task_desc))
        tokens = self.tokenizer(task_desc, return_tensors='pt', truncation=True, max_length=128)
        # tokens = {key: value.to(self.device) for key, value in tokens.items()}
        text_emb = self.text_proj(self.text_encoder(**tokens.to("cuda")).text_embeds)
        # means3D = means3D[None,:,:]
        pcd_emb = self.pcd_encoder(means3D.transpose(2, 1))

        # print(text_emb.size())
        # print(pcd_emb.size())
        x = self.mha1(text_emb, pcd_emb, pcd_emb)
        x = self.drop1(self.fc1(x))
        x = self.drop2(self.fc2(x.reshape(3, 2048)))
        x = self.drop3(self.fc3(x))
        # x = self.drop4(self.fc4(x))
        x = torch.mul(mask, self.drop4(self.fc4(x)))
        x = x.reshape(150000, 3)
        # print(G._xyz.size())
        # print(x.size())
        means3D += x

        if inference:
            return means3D, mask, opacity, scales, rotations, shs
        
        rendered_img = self.render(
            means3D[0], opacity[0], scales[0], rotations[0], shs[0], active_sh_degree[0],
            torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device),
            FovX,
            FovY,
            world_view_transform,
            full_proj_transform,
            camera_center,
        )

        # print(rendered_img.size())
        return rendered_img

    def training_step(self, batch, batch_idx):
        means3D, mask, opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, task_desc, target = batch
        # print(task_desc, type(task_desc))
        output = self(means3D, mask, opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, task_desc)
        loss = self.compute_loss(output, target[0])
        # save = ToPILImage()(output)
        # save.save("render.png")
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        means3D, mask, opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, task_desc, target = batch
        output = self(means3D, mask, opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, task_desc)
        loss = self.compute_loss(output, target)
        self.log("val_loss", loss)

        return loss
    
    def inference_step(self, means3D, mask, opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, task_desc):
        with torch.no_grad():
            means3D, mask, opacity, scales, rotations, shs = self(means3D, mask, opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, task_desc, inference=True)

        return means3D, mask, opacity, scales, rotations, shs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def compute_loss(self, output, target):
        """Compute loss for training and validation."""
        # print(target.size())
        # print(output.size())
        l1 = l1_loss(output, target)
        ssim_loss = ssim(output, target)
        # gt = ToPILImage()(target)
        # gt.save("gt.png")
        lambda_dssim = 0.2
        
        loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_loss)
        return loss






        


