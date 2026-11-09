"""
Temporal Reasoning and Multi-Camera Association Engine.
Combines visual appearance match score with spatial-temporal transition probabilities.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from src.temporal.camera_topology import CameraTopology


class TemporalReasoningEngine:
    """
    Evaluates temporal and camera consistency for candidate person observations.
    Score = alpha * S_visual + (1 - alpha) * S_spatiotemporal
    """
    def __init__(self, topology: Optional[CameraTopology] = None, visual_weight: float = 0.70):
        self.topology = topology or CameraTopology()
        self.visual_weight = visual_weight

    def evaluate_match(self,
                       visual_similarity: float,
                       cam_query: int,
                       time_query: float,
                       cam_candidate: int,
                       time_candidate: float) -> Tuple[float, Dict[str, float]]:
        """
        Combines visual similarity with spatial-temporal physical travel likelihood.
        """
        st_likelihood = self.topology.compute_spatiotemporal_likelihood(
            cam_src=cam_candidate,
            t_src=time_candidate,
            cam_dst=cam_query,
            t_dst=time_query
        )

        combined_score = self.visual_weight * visual_similarity + (1.0 - self.visual_weight) * st_likelihood
        
        breakdown = {
            "visual_similarity": float(visual_similarity),
            "spatiotemporal_likelihood": float(st_likelihood),
            "combined_score": float(combined_score),
            "time_delta_sec": float(abs(time_query - time_candidate)),
            "cam_transition": f"c{cam_candidate} -> c{cam_query}"
        }
        return float(combined_score), breakdown
