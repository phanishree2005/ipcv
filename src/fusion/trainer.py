"""
Training framework and loss functions for M²-ReID adaptive fusion and baseline models.
Implements Triplet Loss with Batch-Hard mining, Label-Smoothing Cross Entropy, and Gating Entropy Regularization.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from typing import Dict, Any, Tuple
import numpy as np
import yaml
from tqdm import tqdm

from src.reid.base_extractor import normalize_features
from src.fusion.adaptive_fusion import AdaptiveMultiModelFusion
from src.quality.quality_estimator import ImageQualityEstimator
from src.pose.pose_estimator import PoseEstimator


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes pair-wise Euclidean distance matrix: dist[i, j] = ||x_i - y_j||^2."""
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


class TripletLoss(nn.Module):
    """
    Triplet loss with hard positive and hard negative mining (Hermans et al., 2017).
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = inputs.size(0)
        dist = euclidean_dist(inputs, inputs)

        # For each anchor, find the hardest positive and hardest negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        # Hardest positive (maximum distance with same target)
        dist_ap = []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)

        # Hardest negative (minimum distance with different target)
        dist_an = []
        for i in range(n):
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss, dist_ap.mean(), dist_an.mean()


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross entropy loss with label smoothing to prevent overconfidence on noisy Re-ID annotations.
    """
    def __init__(self, num_classes: int, epsilon: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = self.logsoftmax(inputs)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
        loss = (-targets_smooth * log_probs).mean(0).sum()
        return loss


class AdaptiveFusionTrainer:
    """
    Orchestrates the training of the Adaptive Multi-Model Fusion module.
    """
    def __init__(self,
                 fusion_model: AdaptiveMultiModelFusion,
                 num_classes: int = 751,
                 lr: float = 3e-4,
                 weight_decay: float = 1e-4,
                 device: str = "cpu",
                 checkpoint_dir: str = "checkpoints"):
        self.fusion_model = fusion_model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.criterion_ce = CrossEntropyLabelSmooth(num_classes=num_classes)
        self.criterion_triplet = TripletLoss(margin=0.3)
        self.optimizer = AdamW(self.fusion_model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=40, eta_min=1e-6)

        self.quality_estimator = ImageQualityEstimator()
        self.pose_estimator = PoseEstimator().to(device)

    def train_epoch(self,
                    train_loader,
                    extractor_osnet,
                    extractor_trans,
                    extractor_clip,
                    epoch: int,
                    max_batches: int = None) -> Dict[str, float]:
        self.fusion_model.train()
        extractor_osnet.eval()
        extractor_trans.eval()
        extractor_clip.eval()

        total_loss = 0.0
        total_ce = 0.0
        total_tri = 0.0
        total_batches = min(len(train_loader), max_batches) if max_batches else len(train_loader)

        for batch_idx, (imgs, pids, camids, _, _) in enumerate(train_loader):
            if max_batches and batch_idx >= max_batches:
                break
            imgs = imgs.to(self.device)
            pids = pids.to(self.device)
            camids = camids.to(self.device)

            with torch.no_grad():
                f_os = extractor_osnet(imgs, camids)
                f_tr = extractor_trans(imgs, camids)
                f_cl = extractor_clip(imgs, camids)
                q_vec = self.quality_estimator.compute_batch_quality(imgs)
                p_vec = torch.from_numpy(
                    np.stack([self.pose_estimator.extract_pose_feature(img) for img in imgs])
                ).float().to(self.device)

            self.optimizer.zero_grad()

            fused_norm, cls_score, weights, bn_feat = self.fusion_model.forward_train(
                f_os, f_tr, f_cl, q_vec, p_vec
            )

            loss_ce = self.criterion_ce(cls_score, pids)
            loss_tri, _, _ = self.criterion_triplet(fused_norm, pids)
            
            # Gating regularization: encourage non-trivial weight combinations (entropy penalty)
            entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
            loss_reg = -0.05 * entropy # Mildly encourage confidence without collapsing

            loss = loss_ce + loss_tri + loss_reg
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_tri += loss_tri.item()

        self.scheduler.step()

        return {
            "epoch": epoch,
            "loss": total_loss / max(1, total_batches),
            "loss_ce": total_ce / max(1, total_batches),
            "loss_triplet": total_tri / max(1, total_batches),
            "lr": self.optimizer.param_groups[0]["lr"]
        }

    def save_checkpoint(self, filename: str = "adaptive_fusion_best.pth"):
        save_path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.fusion_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, save_path)
        print(f"[CHECKPOINT] Model saved to {save_path}")
