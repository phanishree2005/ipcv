"""
Adaptive Multi-Model Feature Fusion (Primary Research Contribution).
Learns dynamic sample-specific gating weights [w_osnet, w_transreid, w_clip]
conditioned on multi-model embeddings, 7-dim image quality features, and 34-dim pose features.
Ensures softmax(weights) sum to 1.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from src.reid.base_extractor import normalize_features


class GatingNetwork(nn.Module):
    """
    Neural Gating Network that maps concatenated representations,
    quality metrics, and pose keypoints into 3 dynamic attention weights.
    """
    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 3) # 3 model weights: OSNet, TransReID, CLIP
        )

    def forward(self, context_vector: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = self.mlp(context_vector)
        weights = F.softmax(logits / max(temperature, 1e-4), dim=-1)
        return weights


class AdaptiveMultiModelFusion(nn.Module):
    """
    M²-ReID Adaptive Feature Fusion Module.
    Inputs:
      - osnet_feat: (B, 512)
      - transreid_feat: (B, 768)
      - clip_feat: (B, 512)
      - quality_feat: (B, 7) [blur, brightness, contrast, det_conf, bbox_area, aspect_ratio, occlusion]
      - pose_feat: (B, 34) [17 x 2 keypoint coordinates]
    Outputs:
      - fused_embedding: (B, 512) L2-normalized
      - gating_weights: (B, 3) where sum(w) = 1.0
    """
    def __init__(self,
                 osnet_dim: int = 512,
                 transreid_dim: int = 768,
                 clip_dim: int = 512,
                 quality_dim: int = 7,
                 pose_dim: int = 34,
                 fused_dim: int = 512,
                 hidden_dim: int = 256,
                 dropout: float = 0.1,
                 num_classes: int = 751):
        super().__init__()
        self.fused_dim = fused_dim
        self.quality_dim = quality_dim
        self.pose_dim = pose_dim
        self.num_classes = num_classes

        # Linear projection layers to align all models to fused_dim
        self.proj_osnet = nn.Sequential(
            nn.Linear(osnet_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_trans = nn.Sequential(
            nn.Linear(transreid_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_clip = nn.Sequential(
            nn.Linear(clip_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True)
        )

        # Quality & Pose encoders
        self.quality_encoder = nn.Sequential(
            nn.Linear(quality_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32)
        )
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64)
        )

        # Total context dimension for Gating Network
        # 3 * fused_dim + 32 (quality) + 64 (pose)
        total_context_dim = 3 * fused_dim + 32 + 64
        self.gating_network = GatingNetwork(in_features=total_context_dim, hidden_dim=hidden_dim, dropout=dropout)

        # Post-fusion refinement transformer / residual block
        self.refinement = nn.Sequential(
            nn.Linear(fused_dim + 32 + 64, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
            nn.BatchNorm1d(fused_dim)
        )

        # BNNeck for training with Identity Classification
        self.bottleneck = nn.BatchNorm1d(fused_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(fused_dim, num_classes, bias=False) if num_classes > 0 else None

    def forward(self,
                osnet_feat: torch.Tensor,
                transreid_feat: torch.Tensor,
                clip_feat: torch.Tensor,
                quality_feat: Optional[torch.Tensor] = None,
                pose_feat: Optional[torch.Tensor] = None,
                temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for inference and evaluation.
        Returns (fused_embedding, gating_weights).
        """
        B = osnet_feat.size(0)
        device = osnet_feat.device

        # Project features
        p_os = self.proj_osnet(osnet_feat)
        p_tr = self.proj_trans(transreid_feat)
        p_cl = self.proj_clip(clip_feat)

        # Handle optional or default quality / pose vectors
        if quality_feat is None:
            quality_feat = torch.full((B, self.quality_dim), 0.5, device=device)
        if pose_feat is None:
            pose_feat = torch.zeros((B, self.pose_dim), device=device)

        q_enc = self.quality_encoder(quality_feat)
        p_enc = self.pose_encoder(pose_feat)

        # Construct full context vector
        context = torch.cat([p_os, p_tr, p_cl, q_enc, p_enc], dim=-1)

        # Compute adaptive softmax weights: (B, 3)
        weights = self.gating_network(context, temperature=temperature)
        w_os = weights[:, 0].unsqueeze(1)
        w_tr = weights[:, 1].unsqueeze(1)
        w_cl = weights[:, 2].unsqueeze(1)

        # Weighted combination
        fused_init = w_os * p_os + w_tr * p_tr + w_cl * p_cl

        # Refinement conditioning on quality & pose
        refine_input = torch.cat([fused_init, q_enc, p_enc], dim=-1)
        refined = self.refinement(refine_input)
        fused_output = fused_init + refined

        # L2 Normalization
        fused_normalized = normalize_features(fused_output)

        return fused_normalized, weights

    def forward_train(self,
                      osnet_feat: torch.Tensor,
                      transreid_feat: torch.Tensor,
                      clip_feat: torch.Tensor,
                      quality_feat: torch.Tensor,
                      pose_feat: torch.Tensor,
                      temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training with BNNeck and classification head.
        Returns (fused_normalized, cls_score, weights, bn_feat).
        """
        B = osnet_feat.size(0)
        p_os = self.proj_osnet(osnet_feat)
        p_tr = self.proj_trans(transreid_feat)
        p_cl = self.proj_clip(clip_feat)

        q_enc = self.quality_encoder(quality_feat)
        p_enc = self.pose_encoder(pose_feat)

        context = torch.cat([p_os, p_tr, p_cl, q_enc, p_enc], dim=-1)
        weights = self.gating_network(context, temperature=temperature)

        w_os = weights[:, 0].unsqueeze(1)
        w_tr = weights[:, 1].unsqueeze(1)
        w_cl = weights[:, 2].unsqueeze(1)

        fused_init = w_os * p_os + w_tr * p_tr + w_cl * p_cl
        refine_input = torch.cat([fused_init, q_enc, p_enc], dim=-1)
        refined = self.refinement(refine_input)
        fused_raw = fused_init + refined

        fused_normalized = normalize_features(fused_raw)
        bn_feat = self.bottleneck(fused_raw)
        cls_score = self.classifier(bn_feat) if self.classifier is not None else None

        return fused_normalized, cls_score, weights, bn_feat
