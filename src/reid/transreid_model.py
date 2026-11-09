"""
TransReID: Transformer-based Person Re-Identification.
Implementation of ViT-based Re-ID with Side Information Embedding (SIE)
for camera/view awareness and patch self-attention (He et al., ICCV 2021).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.reid.base_extractor import BaseReIDExtractor, normalize_features


class PatchEmbed(nn.Module):
    """Image to Patch Embedding for (H, W) -> N patches."""
    def __init__(self, img_size=(256, 128), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """Multi-Head Self Attention."""
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-Forward Network."""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
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
    """Transformer Encoder Block."""
    def __init__(self, dim, num_heads=12, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransReID(BaseReIDExtractor):
    """
    TransReID: Vision Transformer with Camera Side Information Embedding (SIE).
    Generates 768-dim normalized embedding.
    """
    def __init__(self, img_size=(256, 128), patch_size=16, embedding_dim=768,
                 depth=6, num_heads=12, num_cameras=15, num_classes=751, normalize=True):
        super().__init__(embedding_dim=embedding_dim, normalize=normalize)
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embedding_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embedding_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        # Side Information Embedding (Camera IDs)
        self.camera_embed = nn.Parameter(torch.zeros(num_cameras + 1, embedding_dim))

        self.blocks = nn.ModuleList([
            Block(dim=embedding_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embedding_dim)

        # BNNeck
        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)

        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False) if num_classes > 0 else None

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.camera_embed, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor, cam_ids: torch.Tensor = None) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        # Add camera side information if provided
        if cam_ids is not None:
            cam_ids = torch.clamp(cam_ids, 0, self.camera_embed.shape[0] - 1)
            cam_sie = self.camera_embed[cam_ids].unsqueeze(1)
            x = x + pos_embed + cam_sie
        else:
            x = x + pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_feat = x[:, 0]
        bn_feat = self.bottleneck(cls_feat)
        return bn_feat

    def forward_train(self, x: torch.Tensor, cam_ids: torch.Tensor = None):
        feat = self.forward_features(x, cam_ids)
        norm_feat = normalize_features(feat)
        cls_score = self.classifier(feat) if self.classifier is not None else None
        return norm_feat, cls_score
