import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.core import multiarray
from models.pointnet2_semseg import get_model
from models.pointnet2 import PointNet2
from models.pointnet2_cls_ssg import get_model as pn_cls
torch.serialization.add_safe_globals([np])
import open3d as o3d   
from models.gaussian_model import GaussianModel


if __name__ == "__main__":
    # model = get_model(num_classes=13)
    model = pn_cls(num_class=40, normal_channel=False)
    # model = PointNet2(num_class=40, normal_channel=True)
    # model = torch.load("best_model.pth")
    # checkpoint = torch.load("semseg.pth", weights_only=False)
    checkpoint = torch.load("cls_ssg.pth", weights_only=False)
    # checkpoint = torch.load("best_model.pth", weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    # dummy_pcd = torch.randn([2, 100000, 3])
    gt_pcd = torch.tensor(np.asarray(o3d.io.read_point_cloud("/home/ydighe/Developer/datasets/gaussian-splatting/slide_block_to_target/variation_0/episode_0/init_pcd.ply").points).astype(np.float32))
    gt_pcd = gt_pcd[None,:,:]
    splat = GaussianModel(3)
    splat.load_ply("/home/ydighe/Developer/datasets/gaussian-splatting/slide_block_to_target/variation_0/episode_0/splat/point_cloud/iteration_30000/point_cloud.ply")
    splat._xyz.to("cpu")
    splat_pcd = splat.get_xyz
    splat_pcd = splat_pcd[None,:,:].to("cpu")


    with torch.no_grad():
        gt_embedding  = model(gt_pcd.transpose(2,1))
        splat_embedding  = model(splat_pcd.transpose(2,1))

        loss = F.cosine_similarity(gt_embedding, splat_embedding)

    print(loss.item())

    print(splat_pcd)
    splat._xyz = splat._xyz.to("cpu") + torch.ones(splat_pcd.shape)
    print(splat.get_xyz)
    splat._xyz = splat._xyz.to("cpu") + torch.ones(splat_pcd.shape)
    print(splat.get_xyz)
    splat._xyz = splat._xyz.to("cpu")  + torch.ones(splat_pcd.shape)
    print(splat.get_xyz)