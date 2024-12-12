import torch
import torch.nn as nn
import open3d as o3d
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from torchsummary import summary
import lightning as L

from models.gaussian_model import GaussianModel
from models.pointnet2_cls_ssg import PointNet2 as pointnet
from models.espresso import Espresso

from data_scripts.espresso_dataloader import EspressoDataset

from torch.utils.data import DataLoader


def custom_collate_fn(batch):
    """
    Custom collate function to handle batches with mixed types.
    splat, image, "slide red block to green target", world_view_transform, full_proj_transform, camera_center, FovX, FovY
    """
    G_batch = [item[0] for item in batch][0]  # Collect all GaussianModel instances or tensors
    target_batch = [item[1] for item in batch][0]  # Stack target images
    task_desc_batch = [item[2] for item in batch][0]  # Collect all task descriptions
    world_view_transform_batch = [item[3] for item in batch][0]  # Stack world view transforms
    full_proj_transform_batch = [item[4] for item in batch][0]  # Stack full projection transforms
    camera_center_batch = [item[5] for item in batch][0]  # Stack camera centers
    FovX_batch = [item[6] for item in batch][0]  # Stack FovX tensors
    FovY_batch = [item[7] for item in batch][0]  # Stack FovY tensors

    print(task_desc_batch,  type(task_desc_batch))

    return (G_batch, target_batch,task_desc_batch, FovX_batch, FovY_batch, 
            world_view_transform_batch, full_proj_transform_batch, 
            camera_center_batch)

if __name__ == "__main__":
    
    d_model = 512
    # task_desc = "slide red block to green target"
    pretrained_model = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval() 

    print(f"Created text encoder")

    pcd_encoder = pointnet(num_class=40, normal_channel=False)
    checkpoint = torch.load("cls_ssg.pth", weights_only=False)
    pcd_encoder.load_state_dict(checkpoint['model_state_dict'])
    pcd_encoder.eval()
    pcd_encoder.requires_grad_(False)

    pcd_encoder.fc3 = nn.Linear(256, d_model)

    print(f"Created pcd encoder")

    # splat = GaussianModel(3)
    # splat.load_ply("/home/ydighe/Developer/datasets/gaussian-splatting/slide_block_to_target/variation_0/episode_0/splat/point_cloud/iteration_30000/point_cloud.ply")
    # splat_dims = splat.get_xyz.shape

    model = Espresso(tokenizer, text_encoder, pcd_encoder, d_model)

    print(f"Created model")
    # dummy_input = (splat, torch.ones(splat_dims[0]), task_desc)

    train_set = EspressoDataset("train_paths.csv")
    val_set = EspressoDataset("val_paths.csv")

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    print("Obtained train set")
    # splat, image, task_desc ,world_view_transform, full_proj_transform, camera_center, FovX, FovY = next(iter(train_set))
    # summary(model, [splat, task_desc, FovX, FovY, world_view_transform, full_proj_transform, camera_center])
    # summary(model)

    # Trainer
    trainer = L.Trainer(accelerator="cpu", 
                        devices=1,
                        # logger=wandb_logger,
                        max_epochs=50,
                        fast_dev_run=True)

    # Train Model
    trainer.fit(model, train_loader, val_loader)