# Adapted from https://github.com/graphdeco-inria/gaussian-splatting/blob/main/gaussian_renderer/__init__.py

import torch
import torch.nn as nn
import torch
import math

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from models.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

class Renderer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        means3D,            # [B, N, 3]
        opacity,            # [B, N, 1]
        scales,             # [B, N, 3]
        rotations,          # [B, N, 4]
        shs,                # [B, N, ?]
        active_sh_degree,   # [B] or int
        bg_color,           # [B, 3]
        FovX, FovY,         # [B] or float
        world_view_transform,  # [B, 4, 4]
        full_proj_transform,   # [B, 4, 4]
        camera_center,      # [B, 3]
        image_height=128,
        image_width=128,
        scaling_modifier=1.0,
    ):
        B = means3D.shape[0]
        views = world_view_transform[0].shape[0]
        rendered_images = []
        # radii_list = []
        bg = torch.tensor([0,0,0], dtype=torch.float32, device=bg_color.device).unsqueeze(0).repeat(B, 1)
        
        for b in range(B):
            rendered_views = []
            # print(FovX.shape)
            # print(FovX[b].shape)
            # exit(1)
            for view in range(views):
                tanfovx = math.tan(FovX[b][view].item() * 0.5)
                tanfovy = math.tan(FovY[b][view].item() * 0.5)

                raster_settings = GaussianRasterizationSettings(
                    image_height=int(image_height),
                    image_width=int(image_width),
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=bg,  # shape: [3]
                    scale_modifier=scaling_modifier,
                    viewmatrix=world_view_transform[b][view],
                    projmatrix=full_proj_transform[b][view],
                    sh_degree=active_sh_degree[b],
                    campos=camera_center[b][view],
                    prefiltered=False,
                    debug=False
                )

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                screenspace_points = torch.zeros_like(means3D[b], requires_grad=True, device=means3D.device)
                try:
                    screenspace_points.retain_grad()
                except:
                    pass

                rendered_image, _ = rasterizer(
                    means3D=means3D[b],
                    means2D=screenspace_points,
                    shs=shs[b],
                    colors_precomp=None,
                    opacities=opacity[b],
                    scales=scales[b],
                    rotations=rotations[b],
                    cov3D_precomp=None
                )

                rendered_views.append(rendered_image)
                
            rendered_images.append(torch.stack(rendered_views))
            # radii_list.append(radii)  
        
        rendered_images = torch.stack(rendered_images)
        # print(rendered_images.shape)
        # Stack all rendered images back into [B, C, H, W]
        return rendered_images
