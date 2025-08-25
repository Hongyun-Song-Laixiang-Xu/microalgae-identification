import torch
import torch.nn as nn


class BiFormerAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sampling_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.sampling_ratio = sampling_ratio

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # 动态稀疏采样（激活）
        keep_num = int(N * self.sampling_ratio)
        scores = torch.norm(q, dim=-1)
        _, keep_indices = torch.topk(scores, k=keep_num, dim=-1)

        # 稀疏化处理
        k = torch.gather(k, dim=2, index=keep_indices.unsqueeze(-1).expand(-1, -1, -1, k.size(-1)))
        v = torch.gather(v, dim=2, index=keep_indices.unsqueeze(-1).expand(-1, -1, -1, v.size(-1)))

        # 核函数增强的注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.clamp(attn, -50, 50).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return torch.clamp(self.proj(x), -10, 10)