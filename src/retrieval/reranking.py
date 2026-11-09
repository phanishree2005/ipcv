"""
Multi-Modal and Multi-Context Re-Ranking for Person Re-Identification.
Integrates Appearance similarity, Pose consistency, Spatial-Temporal likelihood,
Camera Transition validity, and Image Quality weighting into final candidate ranking.
Also includes k-reciprocal encoding (Zhong et al., CVPR 2017).
"""

import numpy as np
from typing import List, Dict, Any, Optional
from src.pose.pose_estimator import PoseEstimator


def k_reciprocal_reranking(prob_feats: np.ndarray,
                           gal_feats: np.ndarray,
                           k1: int = 20,
                           k2: int = 6,
                           lambda_value: float = 0.3) -> np.ndarray:
    """
    Standard k-reciprocal feature re-ranking (Zhong et al., CVPR 2017).
    Computes Jaccard distance between k-reciprocal nearest neighbors.
    """
    query_num = prob_feats.shape[0]
    all_num = query_num + gal_feats.shape[0]
    feat = np.concatenate([prob_feats, gal_feats], axis=0)

    # Cosine distance matrix
    dist_mat = 1.0 - np.dot(feat, feat.T)
    dist_mat = np.clip(dist_mat, 0.0, 2.0)

    initial_rank = np.argsort(dist_mat, axis=1)

    nn_k1 = []
    for i in range(all_num):
        # Forward k1 neighbors
        forward_k1 = initial_rank[i, :k1 + 1]
        # Backward check
        backward_k1 = initial_rank[forward_k1, :k1 + 1]
        reciprocal_mask = np.isin(backward_k1, i).any(axis=1)
        reciprocal_k1 = forward_k1[reciprocal_mask]
        nn_k1.append(reciprocal_k1)

    # Jaccard distance matrix
    v = np.zeros((all_num, all_num), dtype=np.float32)
    for i in range(all_num):
        v[i, nn_k1[i]] = 1.0

    inv_index = np.dot(v, v.T)
    jaccard_dist = 1.0 - inv_index / (np.sum(v, axis=1, keepdims=True) + np.sum(v, axis=1) - inv_index + 1e-8)

    # Final combined distance
    final_dist = (1 - lambda_value) * dist_mat[:query_num, query_num:] + lambda_value * jaccard_dist[:query_num, query_num:]
    return final_dist


class MultiContextReRanker:
    """
    Configurable composite scoring function for FAISS candidate re-ranking.
    Final Score = w_app * S_app + w_pose * S_pose + w_temp * S_temporal + w_cam * S_camera + w_qual * S_quality
    """
    def __init__(self,
                 w_appearance: float = 0.55,
                 w_pose: float = 0.15,
                 w_temporal: float = 0.15,
                 w_camera: float = 0.10,
                 w_quality: float = 0.05,
                 camera_transition_matrix: Optional[Dict[str, Dict[str, float]]] = None):
        self.w_app = w_appearance
        self.w_pose = w_pose
        self.w_temp = w_temporal
        self.w_cam = w_camera
        self.w_qual = w_quality
        self.transition_matrix = camera_transition_matrix or {}

    def compute_temporal_consistency(self, dt: float, min_time: float = 2.0, max_time: float = 600.0) -> float:
        """
        Gaussian-decay spatial-temporal travel consistency score.
        1.0 if reasonable physical travel time, decays smoothly if too long or zero.
        """
        abs_dt = abs(dt)
        if abs_dt < min_time:
            # Same instant in different camera -> suspicious/distractor unless adjacent wide-angle
            return 0.3
        if abs_dt > max_time:
            return float(np.exp(-(abs_dt - max_time) / 300.0) * 0.5)
        # Optimal travel window
        return 1.0

    def compute_camera_transition_prior(self, cam_q: int, cam_g: int) -> float:
        """Returns prior probability of transit from cam_q to cam_g."""
        if cam_q == cam_g:
            return 0.7 # Same camera match
        q_key = f"c{cam_q}"
        g_key = f"c{cam_g}"
        if q_key in self.transition_matrix and g_key in self.transition_matrix[q_key]:
            return float(self.transition_matrix[q_key][g_key])
        return 0.5 # Default unconstrained prior

    def rerank_candidates(self,
                          query_info: Dict[str, Any],
                          candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-scores and re-sorts FAISS candidates using all multi-context signals.
        """
        reranked = []
        q_cam = query_info.get("camera_id", 1)
        q_time = query_info.get("timestamp", 0.0)
        q_pose = query_info.get("pose_vector")

        for cand in candidates:
            # 1. Appearance Score (FAISS Cosine Similarity in [0, 1])
            s_app = cand.get("similarity", 0.5)
            
            # 2. Pose Similarity
            s_pose = 0.5
            c_pose = cand.get("metadata", {}).get("pose_vector") or cand.get("pose_vector")
            if q_pose is not None and c_pose is not None:
                p1 = np.array(q_pose, dtype=np.float32)
                p2 = np.array(c_pose, dtype=np.float32)
                s_pose = PoseEstimator.compute_pose_similarity(p1, p2)

            # 3. Temporal Consistency
            c_time = cand.get("timestamp", q_time)
            dt = c_time - q_time
            s_temp = self.compute_temporal_consistency(dt)

            # 4. Camera Transition Prior
            c_cam = cand.get("camera_id", q_cam)
            s_cam = self.compute_camera_transition_prior(q_cam, c_cam)

            # 5. Image Quality Score
            s_qual = cand.get("quality_score", 0.8)

            # Composite Score
            final_score = (
                self.w_app * s_app +
                self.w_pose * s_pose +
                self.w_temp * s_temp +
                self.w_cam * s_cam +
                self.w_qual * s_qual
            )

            cand_updated = dict(cand)
            cand_updated["final_score"] = float(final_score)
            cand_updated["scores_breakdown"] = {
                "appearance": float(s_app),
                "pose": float(s_pose),
                "temporal": float(s_temp),
                "camera": float(s_cam),
                "quality": float(s_qual)
            }
            reranked.append(cand_updated)

        # Sort descending by final_score
        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        for i, c in enumerate(reranked, start=1):
            c["reranked_rank"] = i

        return reranked
