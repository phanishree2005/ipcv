"""
Base feature extractor interface and normalization utilities for Person Re-ID.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union, List

def normalize_features(features: Union[torch.Tensor, np.ndarray], eps: float = 1e-12) -> Union[torch.Tensor, np.ndarray]:
    """
    L2 feature normalization. Supports both PyTorch Tensors and NumPy arrays.
    """
    if isinstance(features, torch.Tensor):
        norm = torch.norm(features, p=2, dim=-1, keepdim=True).clamp(min=eps)
        return features / norm
    elif isinstance(features, np.ndarray):
        norm = np.linalg.norm(features, axis=-1, keepdims=True)
        norm = np.maximum(norm, eps)
        return features / norm
    else:
        raise TypeError(f"Unsupported features type: {type(features)}")


class BaseReIDExtractor(nn.Module, ABC):
    """
    Abstract base class for Re-ID feature extractors.
    """
    def __init__(self, embedding_dim: int, normalize: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.normalize = normalize

    @abstractmethod
    def forward_features(self, x: torch.Tensor, cam_ids: torch.Tensor = None) -> torch.Tensor:
        """Extract unnormalized or normalized representation."""
        pass

    def forward(self, x: torch.Tensor, cam_ids: torch.Tensor = None) -> torch.Tensor:
        feat = self.forward_features(x, cam_ids)
        if self.normalize:
            feat = normalize_features(feat)
        return feat
