"""
Model factory for loading and configuring OSNet, TransReID, and CLIP extractors.
"""

from typing import Dict, Any, Union
import torch
import yaml
from pathlib import Path

from src.reid.base_extractor import BaseReIDExtractor, normalize_features
from src.reid.osnet_model import OSNet
from src.reid.transreid_model import TransReID
from src.reid.clip_model import CLIPVisionExtractor

def build_reid_model(model_name: str, config: Dict[str, Any] = None, num_classes: int = 751, device: str = "cpu") -> BaseReIDExtractor:
    """
    Builds a person Re-ID model based on model name and configuration.
    """
    model_name = model_name.lower()
    
    if model_name == "osnet":
        emb_dim = 512 if config is None else config.get("embedding_dim", 512)
        model = OSNet(embedding_dim=emb_dim, num_classes=num_classes)
    elif model_name in ["transreid", "transformer"]:
        emb_dim = 768 if config is None else config.get("embedding_dim", 768)
        model = TransReID(embedding_dim=emb_dim, num_classes=num_classes)
    elif model_name == "clip":
        emb_dim = 512 if config is None else config.get("embedding_dim", 512)
        model = CLIPVisionExtractor(embedding_dim=emb_dim)
    else:
        raise ValueError(f"Unknown Re-ID model name: '{model_name}'. Choose from 'osnet', 'transreid', 'clip'.")

    model = model.to(device)
    return model


class MultiModelFeatureExtractor:
    """
    Unified extractor that extracts OSNet (512d), TransReID (768d), and CLIP (512d)
    embeddings in a single pass.
    """
    def __init__(self, device: str = "cpu", num_classes: int = 751):
        self.device = device
        self.osnet = build_reid_model("osnet", num_classes=num_classes, device=device)
        self.transreid = build_reid_model("transreid", num_classes=num_classes, device=device)
        self.clip = build_reid_model("clip", num_classes=num_classes, device=device)

        self.osnet.eval()
        self.transreid.eval()
        self.clip.eval()

    @torch.no_grad()
    def extract_all(self, img_tensor: torch.Tensor, cam_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Returns normalized embeddings dictionary:
        {
          'osnet': (B, 512),
          'transreid': (B, 768),
          'clip': (B, 512)
        }
        """
        img_tensor = img_tensor.to(self.device)
        if cam_ids is not None:
            cam_ids = cam_ids.to(self.device)

        feat_osnet = self.osnet(img_tensor, cam_ids)
        feat_trans = self.transreid(img_tensor, cam_ids)
        feat_clip = self.clip(img_tensor, cam_ids)

        return {
            "osnet": feat_osnet,
            "transreid": feat_trans,
            "clip": feat_clip
        }
