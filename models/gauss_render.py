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
            means3D, opacity, scales, rotations, shs, active_sh_degree, 
            bg_color : torch.Tensor,
            FovX, FovY, 
            world_view_transform,
            full_proj_transform,
            camera_center,
            image_height = 128,
            image_width = 128, 
            scaling_modifier = 1.0, 
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