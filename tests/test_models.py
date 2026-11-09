"""
Unit tests for Re-ID model backbones (OSNet, TransReID, CLIP).
"""

import pytest
import torch
from src.reid.osnet_model import OSNet
from src.reid.transreid_model import TransReID
from src.reid.clip_model import CLIPVisionExtractor
from src.reid.base_extractor import normalize_features

def test_osnet_forward():
    model = OSNet(embedding_dim=512, num_classes=10)
    x = torch.randn(2, 3, 256, 128)
    feat = model(x)
    assert feat.shape == (2, 512)
    # Check L2 norm equals 1.0
    norms = torch.norm(feat, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_transreid_forward():
    model = TransReID(embedding_dim=768, num_classes=10)
    x = torch.randn(2, 3, 256, 128)
    cams = torch.tensor([1, 2])
    feat = model(x, cams)
    assert feat.shape == (2, 768)
    norms = torch.norm(feat, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_clip_forward():
    model = CLIPVisionExtractor(embedding_dim=512)
    x = torch.randn(2, 3, 224, 224)
    feat = model(x)
    assert feat.shape == (2, 512)
    norms = torch.norm(feat, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
