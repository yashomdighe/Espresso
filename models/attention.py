import torch
import torch.nn as nn
import copy
import math

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        # Define the 4 projection matrices WQ, WK, WV & WO for attention and concat
        self.proj_matrices = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        self.drop_out = nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2) 
            for lin, x in zip(self.proj_matrices, (query, key, value))
        ]

        # Scaled dot product attention
        dk1 = query.size(-1)
        print(f"d_k: {self.d_k}, dk1: {dk1}")
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        if self.drop_out is not None:
            p_attn = self.drop_out(p_attn)
        
        x = torch.matmul(p_attn, value)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )

        return self.proj_matrices[-1](x)



        
