"""
Unit tests for Fixed and Adaptive Multi-Model Feature Fusion.
"""

import pytest
import torch
from src.fusion.fixed_fusion import FixedFeatureFusion
from src.fusion.adaptive_fusion import AdaptiveMultiModelFusion
from src.fusion.trainer import TripletLoss, CrossEntropyLabelSmooth

def test_fixed_fusion():
    fusion = FixedFeatureFusion(weights=[0.4, 0.4, 0.2], fused_dim=512)
    f_os = torch.randn(4, 512)
    f_tr = torch.randn(4, 768)
    f_cl = torch.randn(4, 512)

    fused = fusion(f_os, f_tr, f_cl)
    assert fused.shape == (4, 512)
    norms = torch.norm(fused, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_adaptive_fusion_gating_sum():
    fusion = AdaptiveMultiModelFusion(fused_dim=512, num_classes=20)
    f_os = torch.randn(4, 512)
    f_tr = torch.randn(4, 768)
    f_cl = torch.randn(4, 512)
    q_vec = torch.randn(4, 7)
    p_vec = torch.randn(4, 34)

    fused, weights = fusion(f_os, f_tr, f_cl, q_vec, p_vec)
    assert fused.shape == (4, 512)
    assert weights.shape == (4, 3)

    # Gating weights must sum to 1.0
    weight_sums = torch.sum(weights, dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

def test_adaptive_fusion_loss_backward():
    fusion = AdaptiveMultiModelFusion(fused_dim=512, num_classes=10)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=1e-3)
    loss_ce = CrossEntropyLabelSmooth(num_classes=10)
    loss_tri = TripletLoss(margin=0.3)

    f_os = torch.randn(8, 512)
    f_tr = torch.randn(8, 768)
    f_cl = torch.randn(8, 512)
    q_vec = torch.randn(8, 7)
    p_vec = torch.randn(8, 34)
    targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    optimizer.zero_grad()
    fused_norm, cls_score, weights, _ = fusion.forward_train(f_os, f_tr, f_cl, q_vec, p_vec)
    l1 = loss_ce(cls_score, targets)
    l2, _, _ = loss_tri(fused_norm, targets)
    total_loss = l1 + l2
    total_loss.backward()
    optimizer.step()

    assert total_loss.item() > 0.0
