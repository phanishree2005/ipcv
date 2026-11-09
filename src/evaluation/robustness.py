"""
Robustness and Perturbation Evaluation Module for Person Re-Identification.
Applies controlled visual degradations:
- Gaussian Blur
- Illumination / Brightness shifts
- Contrast reduction / degradation
- Synthetic Occlusions
- Low Resolution downsampling
Evaluates model degradation curves under real-world domain shifts.
"""

import numpy as np
import cv2
import torch
from typing import Dict, Any, List, Tuple
from PIL import Image

class RobustnessEvaluator:
    """
    Applies synthetic perturbations to query/gallery images and tests Re-ID resilience.
    """
    @staticmethod
    def apply_blur(img: np.ndarray, severity: int = 2) -> np.ndarray:
        """Severity 1 to 5 maps to kernel sizes 3 to 11."""
        k = 2 * severity + 1
        return cv2.GaussianBlur(img, (k, k), 0)

    @staticmethod
    def apply_brightness(img: np.ndarray, factor: float = 0.5) -> np.ndarray:
        """Scales pixel intensity."""
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_contrast(img: np.ndarray, factor: float = 0.5) -> np.ndarray:
        """Scales contrast around 128 mid-point."""
        mean = 128.0
        return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_occlusion(img: np.ndarray, occlude_ratio: float = 0.25) -> np.ndarray:
        """Places a synthetic occluding black/gray block over the crop."""
        out = img.copy()
        h, w = img.shape[:2]
        occ_h = int(h * np.sqrt(occlude_ratio))
        occ_w = int(w * np.sqrt(occlude_ratio))

        y1 = np.random.randint(0, max(1, h - occ_h))
        x1 = np.random.randint(0, max(1, w - occ_w))
        out[y1:y1 + occ_h, x1:x1 + occ_w] = 120 # Gray occluder
        return out

    @staticmethod
    def apply_low_resolution(img: np.ndarray, downsample_factor: int = 4) -> np.ndarray:
        """Downsamples image to simulate distant low-resolution cameras."""
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(8, w // downsample_factor), max(16, h // downsample_factor)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    def generate_perturbed_batch(self, imgs_np: List[np.ndarray], perturbation_type: str, severity: float) -> List[np.ndarray]:
        perturbed = []
        for img in imgs_np:
            if perturbation_type == "blur":
                p_img = self.apply_blur(img, severity=int(severity))
            elif perturbation_type == "brightness_low":
                p_img = self.apply_brightness(img, factor=float(severity))
            elif perturbation_type == "brightness_high":
                p_img = self.apply_brightness(img, factor=float(severity))
            elif perturbation_type == "contrast":
                p_img = self.apply_contrast(img, factor=float(severity))
            elif perturbation_type == "occlusion":
                p_img = self.apply_occlusion(img, occlude_ratio=float(severity))
            elif perturbation_type == "low_res":
                p_img = self.apply_low_resolution(img, downsample_factor=int(severity))
            else:
                p_img = img
            perturbed.append(p_img)
        return perturbed
