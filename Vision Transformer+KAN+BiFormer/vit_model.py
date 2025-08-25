"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from biformer import BiFormerAttention
from kan_layers import KANBlock
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


class ViTWithBiFormer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12):
        super().__init__()

        # 1. 图像分块嵌入
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 2. 计算分块数量
        num_patches = (img_size // patch_size) ** 2

        # 3. 位置编码和CLS Token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 4. Transformer Blocks (使用ModuleList代替ModuleDict)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                BiFormerAttention(embed_dim, num_heads=num_heads),  # 假设已定义BiFormerAttention
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim)
            ) for _ in range(depth)
        ])

        # 5. 分类头
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # 分块嵌入
        x = self.patch_embed(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # 添加CLS Token和位置编码
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
        x = x + self.pos_embed

        # 通过Transformer Blocks
        for block in self.blocks:
            x = x + block(x)  # 残差连接

        # 分类
        cls_output = x[:, 0]  # 取CLS Token
        return self.head(cls_output)
#----------------------------------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_kan=False,
                 use_biformer=False,
                 layer_type='middle',  # 新增参数：'early'/'middle'/'late'
                 **kwargs):
        super().__init__()
        #self.use_kan = use_kan
        self.use_biformer = use_biformer
        self.layer_type = layer_type

        # 根据层级类型动态调整配置
        if layer_type == 'early':
            # 浅层：强化BiFormer，简化KAN
            attn_heads = num_heads
            kan_expansion = 2  # 较窄的KAN
        elif layer_type == 'late':
            # 深层：强化KAN，简化注意力
            attn_heads = max(1, num_heads // 2)  # 减少注意力头
            kan_expansion = 6  # 更宽的KAN
        else:
            # 中层：平衡配置
            attn_heads = num_heads
            kan_expansion = 4

        # 前置归一化层
        self.norm_pre = norm_layer(dim)

        # 注意力模块
        self.norm1 = norm_layer(dim)
        if use_biformer:
            self.attn = BiFormerAttention(
                dim=dim,
                num_heads=attn_heads,
                attn_drop=attn_drop_ratio,
                proj_drop=drop_ratio,
                qkv_bias=qkv_bias
            )
        else:
            self.attn = Attention(
                dim=dim,
                num_heads=attn_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop_ratio=attn_drop_ratio,
                proj_drop_ratio=drop_ratio
            )

        # MLP/KAN 模块
        self.norm2 = norm_layer(dim)
        if use_kan:
            self.mlp = KANBlock(
                dim=dim,
                expansion=kan_expansion,
                act_layer=act_layer,
                drop=drop_ratio
            )
        else:
         mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop_ratio
            )

        # 随机深度
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, x):
        # 前置归一化
        x_norm = self.norm_pre(x)

        if self.layer_type == 'late':
            # 深层：减弱注意力，增强KAN
            attn_out = self.attn(x_norm)
            attn_out = torch.clamp(attn_out, -10.0, 10.0)  # 稳定训练
            x = x + 0.3 * self.drop_path(attn_out)

            mlp_out = self.mlp(self.norm2(x))
            mlp_out = torch.clamp(mlp_out, -5.0, 5.0)
            x = x + 0.7 * self.drop_path(mlp_out)
        else:
            # 浅层和中层：标准残差连接
            attn_out = self.attn(x_norm)
            attn_out = torch.clamp(attn_out, -10.0, 10.0)  # 稳定训练
            x = x + self.drop_path(attn_out)

            mlp_out = self.mlp(self.norm2(x))
            mlp_out = torch.clamp(mlp_out, -5.0, 5.0)
            x = x + self.drop_path(mlp_out)

        return x

    def _init_weights(self):
        # 注意力层初始化（浅层更大初始化）
        if hasattr(self.attn, 'qkv'):
            if self.layer_type == 'early':
                nn.init.xavier_uniform_(self.attn.qkv.weight, gain=1.0)
            else:
                nn.init.xavier_uniform_(self.attn.qkv.weight, gain=0.8)
            if self.attn.qkv.bias is not None:
                nn.init.zeros_(self.attn.qkv.bias)
        if hasattr(self.attn, 'proj'):
            if self.layer_type == 'early':
                nn.init.xavier_uniform_(self.attn.proj.weight, gain=1.0)
            else:
                nn.init.xavier_uniform_(self.attn.proj.weight, gain=0.8)

        # KAN层初始化（深层更小初始化）
        if hasattr(self.mlp, 'coeff'):
            if self.layer_type == 'late':
                nn.init.normal_(self.mlp.coeff, std=0.02)
            else:
                nn.init.kaiming_normal_(self.mlp.coeff, mode='fan_in')

        # MLP层初始化
        if hasattr(self.mlp, 'fc1'):
            nn.init.xavier_uniform_(self.mlp.fc1.weight)
        if hasattr(self.mlp, 'fc2'):
            nn.init.xavier_uniform_(self.mlp.fc2.weight)

    # 在Block类中优化forward方法
    def forward(self, x):
        # 前置归一化
        x_norm = self.norm_pre(x)

        # BiFormer注意力分支
        attn_out = self.attn(x_norm)
        attn_out = torch.clamp(attn_out, -10.0, 10.0)  # 稳定训练

        # KAN/MLP分支
        mlp_out = self.mlp(self.norm2(x + attn_out))  # 注意这里的顺序调整
        mlp_out = torch.clamp(mlp_out, -5.0, 5.0)

        # 残差连接
        x = x + self.drop_path(attn_out) + self.drop_path(mlp_out)
        return x

    def get_attention_map(self, x):
        """获取注意力图（用于可视化）"""
        if hasattr(self.attn, 'get_attention_map'):
            return self.attn.get_attention_map(self.norm1(x))
        return None


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, use_biformer=False, use_kan=False):
        super().__init__()
        self.use_kan = use_kan
        self.use_biformer = use_biformer
        self._debug_mode = False  # 新增调试开关

        # 参数统计方法
        def count_parameters(model):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return total_params, trainable_params

        # 模块参数统计
        def module_parameters(model):
            module_counts = {}
            for name, module in model.named_children():
                module_counts[name] = sum(p.numel() for p in module.parameters())
            return module_counts
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """

        # 原有初始化代码
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # 修改blocks的初始化
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                  attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer, use_biformer=use_biformer, layer_type='early' if i < depth//3 else 'middle' if i < 2*depth//3 else 'late')
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def set_debug(self, mode=True):
        """启用/禁用调试输出"""
        self._debug_mode = mode

    def _print_layer_ranges(self, x):
        """打印各层数值范围"""
        print("\n===== 数值范围验证 =====")
        for i, blk in enumerate(self.blocks[:1]):  # 只检查第一个block
            # 注意力范围
            attn_map = blk.attn.get_attention_map(blk.norm1(x))
            print(f"Block {i} 注意力分数范围: {attn_map.min():.3f}~{attn_map.max():.3f}")

            # KAN/MLP范围
            mlp_out = blk.mlp(blk.norm2(x))
            print(f"Block {i} MLP输出范围: {mlp_out.min():.3f}~{mlp_out.max():.3f}")

            # 残差连接后范围
            x = blk(x)
            print(f"Block {i} 输出范围: {x.min():.3f}~{x.max():.3f}")
            print("-" * 50)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model




def vit_base_patch16_224_in21k(num_classes=21843, has_logits=True, use_biformer=False,use_kan=False):
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=6,      #12,#减少层数
            num_heads=12,
            representation_size=768 if has_logits else None,
            num_classes=num_classes,
            use_kan=use_kan,  # 新增参数
            use_biformer=use_biformer  # 新增参数
           )
        return model


def vit_base_patch32_224(num_classes: int = 1000, use_biformer: bool = False):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes,
                              use_biformer=use_biformer
                               )
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
