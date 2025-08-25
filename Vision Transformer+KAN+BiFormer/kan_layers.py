import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=3):
        super().__init__()
        self.degree = degree
        self.coeff = nn.Parameter(torch.randn(output_dim, degree+1, input_dim))
        nn.init.kaiming_normal_(self.coeff, mode='fan_in')

    def forward(self, x):
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, features = x.shape
            x = x.reshape(-1, features)

        if x.dim() != 2:
            raise ValueError(f"输入张量x的维度应为2或3，但实际为{original_shape}")

        # 输入裁剪
        x = torch.clamp(x, -5, 5)
        x = x.unsqueeze(-1)  # [..., input_dim, 1]
        powers = torch.arange(self.degree + 1, device=x.device)
        x_pow = torch.pow(x, powers)  # [..., input_dim, degree+1]

        # 安全计算（防止NaN）
        x_pow = torch.nan_to_num(x_pow, nan=0.0, posinf=1e4, neginf=-1e4)
        coeff = torch.clamp(self.coeff, -1, 1)  # 限制系数范围

        # 修改einsum表达式
        result = torch.einsum('...dk,okd->...o', x_pow, self.coeff)

        if len(original_shape) == 3:
            result = result.reshape(batch_size, seq_len, -1)
        return result * 0.1  # 缩小输出幅度


class KANBlock(nn.Module):
    def __init__(self, dim, expansion=4, degree=3, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = int(dim * expansion)
        self.kan1 = KANLayer(dim, self.hidden_dim, degree)
        self.kan2 = KANLayer(self.hidden_dim, dim, degree)
        self.norm = nn.LayerNorm(dim)
        self.act_layer = act_layer  # 存储激活层类
        self.activation = self.act_layer()  # 创建激活层实例
        self.dropout = nn.Dropout(drop)  # 添加 Dropout 层

    def forward(self, x):
        # 保存原始形状用于残差连接
        original_shape = x.shape
        residual = x

        # 处理3D输入 (batch, seq_len, features)
        if x.dim() == 3:
            batch_size, seq_len, features = x.shape
            x = x.reshape(-1, features)

            # 前向传播
            x = self.kan1(x)
            x = torch.clamp(x, -5, 5)  # 限制中间输出范围
            x = self.activation(x)  # 使用定义的激活层
            x = self.dropout(x)  # 应用 Dropout
            x = self.kan2(x)
            x = torch.clamp(x, -5, 5)  # 限制中间输出范围
            x = self.dropout(x)  # 应用 Dropout

        # 恢复形状并添加残差
        if len(original_shape) == 3:
            x = x.reshape(batch_size, seq_len, -1)

        return self.norm(x + residual)