"""
Fixed Multi-Model Feature Fusion for Person Re-Identification.
Fuses OSNet, TransReID, and CLIP embeddings via fixed weights with L2 normalization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union
from src.reid.base_extractor import normalize_features


class FixedFeatureFusion(nn.Module):
    """
    Combines representations using fixed linear weighting followed by projection or normalization:
    f_fixed = normalize(w_osnet * f_osnet_proj + w_trans * f_trans_proj + w_clip * f_clip_proj)
    """
    def __init__(self, weights: List[float] = None,
                 osnet_dim: int = 512, transreid_dim: int = 768, clip_dim: int = 512,
                 fused_dim: int = 512):
        super().__init__()
        if weights is None:
            weights = [0.40, 0.40, 0.20] # Default balanced configuration
        
        # Normalize weights to sum to 1.0
        w_sum = sum(weights)
        self.weights = [w / w_sum for w in weights]
        
        self.proj_osnet = nn.Linear(osnet_dim, fused_dim, bias=False) if osnet_dim != fused_dim else nn.Identity()
        self.proj_trans = nn.Linear(transreid_dim, fused_dim, bias=False) if transreid_dim != fused_dim else nn.Identity()
        self.proj_clip = nn.Linear(clip_dim, fused_dim, bias=False) if clip_dim != fused_dim else nn.Identity()

    def forward(self, osnet_feat: torch.Tensor, trans_feat: torch.Tensor, clip_feat: torch.Tensor) -> torch.Tensor:
        # Ensure individual features are normalized
        f_os = normalize_features(self.proj_osnet(osnet_feat))
        f_tr = normalize_features(self.proj_trans(trans_feat))
        f_cl = normalize_features(self.proj_clip(clip_feat))

        fused = (self.weights[0] * f_os +
                 self.weights[1] * f_tr +
                 self.weights[2] * f_cl)
        
        return normalize_features(fused)

    @staticmethod
    def fuse_numpy(osnet_emb: np.ndarray, trans_emb: np.ndarray, clip_emb: np.ndarray,
                   weights: List[float] = (0.40, 0.40, 0.20)) -> np.ndarray:
        """NumPy implementation for fast evaluation during inference."""
        w = np.array(weights, dtype=np.float32)
        w = w / np.sum(w)

        # Truncate or pad if dimension mismatch for direct concatenation / weighted sum
        min_dim = min(osnet_emb.shape[-1], trans_emb.shape[-1], clip_emb.shape[-1])
        f_os = osnet_emb[..., :min_dim]
        f_tr = trans_emb[..., :min_dim]
        f_cl = clip_emb[..., :min_dim]

        f_os = normalize_features(f_os)
        f_tr = normalize_features(f_tr)
        f_cl = normalize_features(f_cl)

        fused = w[0] * f_os + w[1] * f_tr + w[2] * f_cl
        return normalize_features(fused)
