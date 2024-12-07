import torch
import torch.nn as nn

from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPTextModelWithProjection
from models.attention import MultiHeadedAttention
from models.pointnet2 import PointNet2

import open3d as o3d
import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

# class TextEncoder(nn.module):
#     def __init__(self, clip_model, n_dims):
#         self.clip_encoder = clip_model

if __name__ == "__main__":

    pretrained_model = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    # text_encoder = CLIPModel.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    # Declare gaussian model
    # Get means using get_xyz function
    tokens = tokenizer(["slide red block to green target"]*2, return_tensors = 'pt', truncation = True, max_length=128)
    # print(**tokens["size"])
    text_emb = text_encoder(**tokens).text_embeds
    d_model = 512
    proj = nn.Linear(768, 512)
    mha1 = MultiHeadedAttention(h = 8, d_model= d_model, dropout = 0.1)
    mha2 = MultiHeadedAttention(h = 8, d_model= d_model, dropout = 0.1)
    # foo = torch.randn(2, 1, d_model)


    dummy_pcd = torch.randn([2, 100000, 3])
    # dummy_pcd2 = torch.randn([100000, 3])
    # dummy_pcd = torch.cat([torch.randn([100000, 3]),torch.randn([100000, 3])])
    # test_pcd = torch.tensor([np.asarray(o3d.io.read_point_cloud("/home/ydighe/Developer/Espresso/23.ply").points).astype(np.float32)])
    # print(test_pcd.shape)
    pointnet = PointNet2(emb_dim=d_model, normal_channel=False)
    pcd_emb = pointnet(dummy_pcd.transpose(2,1)).view(-1, 1, 512)
    print(f"Text embeddings_size: {text_emb.size()}")
    print(f"PCD embeddings_size: {pcd_emb.size()}")
    # e1 = mha2(pcd_emb, pcd_emb, text_emb)
    proj_text = proj(text_emb)
    print(f"proj_text: {proj_text.size()}")
    e1 = mha2(pcd_emb, pcd_emb, proj_text)
    e2 = mha2(e1, e1, e1)