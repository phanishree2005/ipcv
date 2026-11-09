"""
CLIP Visual Representation Extractor for Person Re-Identification.
Extracts complementary visual features using a Vision-Language foundation model visual encoder.
Identified explicitly as a complementary visual representation (not an inherently specialized Re-ID model).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.reid.base_extractor import BaseReIDExtractor, normalize_features


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        norm_x = self.ln_1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = self.resblocks(x)
        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIPVisionExtractor(BaseReIDExtractor):
    """
    CLIP Visual Encoder for complementary person visual representation.
    Produces a 512-dimensional normalized embedding.
    """
    def __init__(self, embedding_dim: int = 512, input_size: int = 224, normalize: bool = True):
        super().__init__(embedding_dim=embedding_dim, normalize=normalize)
        self.input_size = input_size
        self.visual = VisualTransformer(
            input_resolution=input_size,
            patch_size=16,
            width=768,
            layers=6,
            heads=12,
            output_dim=embedding_dim
        )
        self.resize = nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=False)

    def forward_features(self, x: torch.Tensor, cam_ids: torch.Tensor = None) -> torch.Tensor:
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            x = self.resize(x)
        feat = self.visual(x)
        return feat
