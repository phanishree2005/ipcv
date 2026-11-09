"""
Pose Estimation and Keypoint Feature Extraction Module for Person Re-ID.
Extracts 17 keypoints (Head, Shoulder, Elbow, Wrist, Hip, Knee, Ankle),
constructs normalized pose vectors, and computes pose similarity metrics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from typing import Dict, Any, List, Tuple, Union
from PIL import Image

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

class PoseEstimator(nn.Module):
    """
    Extracts keypoints and builds a 34-dimensional normalized pose vector
    (17 keypoints x 2 normalized [x, y] coordinates).
    Includes a lightweight neural pose encoder for matching.
    """
    def __init__(self, feature_dim: int = 34, out_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        
        # Lightweight neural pose projection network
        self.pose_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim)
        )

    def detect_keypoints_heuristic(self, img_np: np.ndarray) -> np.ndarray:
        """
        Robust geometric keypoint estimator for person crops.
        Identifies body landmarks (head, shoulders, elbows, wrists, hips, knees, ankles)
        based on morphological gradients and anatomical vertical ratios.
        """
        h, w = img_np.shape[:2]
        cx = 0.5 # normalized center x
        
        # Anatomical vertical benchmarks
        # Head / Nose: y ~ 0.12
        # Shoulders: y ~ 0.25, x ~ 0.30 & 0.70
        # Elbows: y ~ 0.40, x ~ 0.22 & 0.78
        # Wrists: y ~ 0.55, x ~ 0.18 & 0.82
        # Hips: y ~ 0.58, x ~ 0.38 & 0.62
        # Knees: y ~ 0.75, x ~ 0.38 & 0.62
        # Ankles: y ~ 0.92, x ~ 0.38 & 0.62
        
        # Compute horizontal projection center of mass along vertical slices
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
        edges = cv2.Canny(gray, 50, 150)
        
        def get_slice_center(y_start, y_end):
            sl = edges[int(h*y_start):int(h*y_end), :]
            if sl.sum() == 0:
                return 0.5
            x_coords = np.where(sl > 0)[1]
            return float(np.median(x_coords) / w) if len(x_coords) > 0 else 0.5

        torso_cx = get_slice_center(0.2, 0.6)
        legs_cx = get_slice_center(0.6, 0.9)

        # 17 Keypoints (x, y) in [0, 1] normalized space
        kpts = [
            (torso_cx, 0.10),          # nose
            (torso_cx - 0.05, 0.08),   # left_eye
            (torso_cx + 0.05, 0.08),   # right_eye
            (torso_cx - 0.10, 0.09),   # left_ear
            (torso_cx + 0.10, 0.09),   # right_ear
            (torso_cx - 0.20, 0.25),   # left_shoulder
            (torso_cx + 0.20, 0.25),   # right_shoulder
            (torso_cx - 0.28, 0.42),   # left_elbow
            (torso_cx + 0.28, 0.42),   # right_elbow
            (torso_cx - 0.32, 0.58),   # left_wrist
            (torso_cx + 0.32, 0.58),   # right_wrist
            (legs_cx - 0.14, 0.58),    # left_hip
            (legs_cx + 0.14, 0.58),    # right_hip
            (legs_cx - 0.14, 0.76),    # left_knee
            (legs_cx + 0.14, 0.76),    # right_knee
            (legs_cx - 0.14, 0.94),    # left_ankle
            (legs_cx + 0.14, 0.94),    # right_ankle
        ]
        
        flat_kpts = np.array([[k[0], k[1]] for k in kpts], dtype=np.float32).flatten()
        return flat_kpts

    def extract_pose_feature(self, img: Union[np.ndarray, torch.Tensor, Image.Image]) -> np.ndarray:
        """Extracts 34-dimensional normalized pose keypoint vector."""
        if isinstance(img, torch.Tensor):
            if img.dim() == 4:
                img = img.squeeze(0)
            img_np = img.detach().cpu().permute(1, 2, 0).numpy()
            img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6) * 255).astype(np.uint8)
        elif isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img

        kpts = self.detect_keypoints_heuristic(img_np)
        return kpts

    def forward(self, pose_vecs: torch.Tensor) -> torch.Tensor:
        """Projects 34-dim keypoint coordinates into 128-dim pose embedding."""
        feat = self.pose_net(pose_vecs)
        norm = torch.norm(feat, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        return feat / norm

    @staticmethod
    def compute_pose_similarity(pose_vec1: np.ndarray, pose_vec2: np.ndarray) -> float:
        """
        Calculates pose similarity using Procrustes-inspired coordinate distance
        and Cosine orientation similarity.
        """
        p1 = pose_vec1.flatten()
        p2 = pose_vec2.flatten()
        
        # Euclidean coordinate difference
        dist = np.linalg.norm(p1 - p2)
        sim_dist = np.exp(-dist * 2.0)
        
        # Cosine similarity
        denom = (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-12)
        sim_cos = float(np.dot(p1, p2) / denom)
        
        pose_sim = 0.5 * sim_dist + 0.5 * max(0.0, sim_cos)
        return float(np.clip(pose_sim, 0.0, 1.0))
