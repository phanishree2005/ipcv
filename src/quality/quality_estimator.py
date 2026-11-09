"""
Image Quality Assessment Module for Person Re-Identification.
Extracts 7-dimensional quality vector:
[blur_laplacian, brightness, contrast, detection_conf, bbox_area_norm, aspect_ratio_deviation, occlusion_score].
Conditioning input for Adaptive Feature Fusion gating.
"""

import numpy as np
import cv2
import torch
from typing import Union, Dict, Any, List
from PIL import Image

class ImageQualityEstimator:
    """
    Computes visual quality indicators for person crops to dynamically modulate
    model weights in adaptive feature fusion.
    """
    def __init__(self):
        self.feature_dim = 7

    def estimate_blur(self, gray_img: np.ndarray) -> float:
        """Calculates variance of Laplacian (higher = sharper, lower = blurrier). Normalized to [0, 1]."""
        laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
        # Non-linear mapping: 0 to 1000+ -> [0, 1]
        score = 1.0 - np.exp(-laplacian_var / 150.0)
        return float(np.clip(score, 0.0, 1.0))

    def estimate_brightness(self, gray_img: np.ndarray) -> float:
        """Measures mean intensity normalized around ideal mid-gray (128)."""
        mean_val = np.mean(gray_img) / 255.0
        # Optimal around 0.5, penalty for extreme dark or blown-out bright
        score = 1.0 - 2.0 * abs(mean_val - 0.5)
        return float(np.clip(score, 0.0, 1.0))

    def estimate_contrast(self, gray_img: np.ndarray) -> float:
        """Measures standard deviation of pixel intensities."""
        std_val = np.std(gray_img) / 128.0
        score = 1.0 - np.exp(-std_val * 2.0)
        return float(np.clip(score, 0.0, 1.0))

    def estimate_aspect_ratio_deviation(self, h: int, w: int, ideal_ratio: float = 2.0) -> float:
        """Evaluates whether crop matches standard standing pedestrian aspect ratio (2:1)."""
        if w == 0:
            return 0.0
        current_ratio = h / float(w)
        deviation = abs(current_ratio - ideal_ratio)
        score = np.exp(-deviation)
        return float(np.clip(score, 0.0, 1.0))

    def estimate_occlusion_heuristic(self, img_bgr: np.ndarray) -> float:
        """
        Heuristic occlusion estimation using edge density and spatial color variance
        across 3 vertical segments (Head, Torso, Legs).
        """
        h, w = img_bgr.shape[:2]
        if h < 30 or w < 15:
            return 0.5
        
        # Segment into 3 parts
        h3 = h // 3
        part1 = img_bgr[0:h3, :]
        part2 = img_bgr[h3:2*h3, :]
        part3 = img_bgr[2*h3:h, :]
        
        # Compare entropy / variance across segments
        vars = [np.var(p) for p in [part1, part2, part3]]
        min_v = min(vars) + 1e-5
        max_v = max(vars) + 1e-5
        ratio = min_v / max_v # Low ratio indicates one segment is occluded/uniform
        occlusion_risk = 1.0 - ratio
        return float(np.clip(occlusion_risk, 0.0, 1.0))

    def extract_quality_vector(self,
                               img: Union[np.ndarray, Image.Image, torch.Tensor],
                               detection_conf: float = 0.90,
                               orig_frame_size: tuple = (1080, 1920)) -> np.ndarray:
        """
        Extracts 7-dimensional normalized quality vector:
        [blur, brightness, contrast, det_conf, bbox_area, aspect_ratio_score, occlusion_score]
        """
        if isinstance(img, torch.Tensor):
            # De-normalize / convert tensor to numpy uint8
            if img.dim() == 4:
                img = img.squeeze(0)
            img_np = img.detach().cpu().permute(1, 2, 0).numpy()
            img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6) * 255).astype(np.uint8)
        elif isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img

        if len(img_np.shape) == 2:
            gray = img_np
            bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        else:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        h, w = gray.shape[:2]
        blur = self.estimate_blur(gray)
        brightness = self.estimate_brightness(gray)
        contrast = self.estimate_contrast(gray)
        det_conf = float(np.clip(detection_conf, 0.0, 1.0))
        
        frame_area = orig_frame_size[0] * orig_frame_size[1]
        crop_area = h * w
        area_norm = float(np.clip(crop_area / (frame_area * 0.1 + 1e-5), 0.0, 1.0))
        
        aspect_score = self.estimate_aspect_ratio_deviation(h, w)
        occlusion = self.estimate_occlusion_heuristic(bgr)

        # Composite overall quality
        quality_vec = np.array([
            blur,
            brightness,
            contrast,
            det_conf,
            area_norm,
            aspect_score,
            occlusion
        ], dtype=np.float32)

        return quality_vec

    def compute_batch_quality(self, img_batch: torch.Tensor, confs: List[float] = None) -> torch.Tensor:
        """Extracts quality vectors for a batch of images."""
        B = img_batch.size(0)
        vecs = []
        for i in range(B):
            conf = confs[i] if confs is not None else 0.90
            q_vec = self.extract_quality_vector(img_batch[i], detection_conf=conf)
            vecs.append(q_vec)
        return torch.from_numpy(np.stack(vecs, axis=0)).float().to(img_batch.device)
