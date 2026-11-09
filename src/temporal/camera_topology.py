"""
Spatial-Temporal Camera Network Topology and Travel Time Modeling.
Models multi-camera adjacency, expected travel time distributions,
and physical transition feasibility.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


class CameraTopology:
    """
    Maintains the camera transition graph and Gaussian travel-time parameters:
    P(Delta t | Cam_i -> Cam_j) ~ N(mu_ij, sigma_ij^2)
    """
    def __init__(self, num_cameras: int = 6, transition_matrix: Optional[Dict[str, Dict[str, float]]] = None):
        self.num_cameras = num_cameras
        self.transition_matrix = transition_matrix or {}
        
        # Default travel time parameters (mean seconds, std seconds) between camera pairs
        self.travel_time_params = {
            (1, 2): (35.0, 15.0),
            (2, 1): (35.0, 15.0),
            (2, 3): (45.0, 20.0),
            (3, 2): (45.0, 20.0),
            (3, 4): (30.0, 12.0),
            (4, 3): (30.0, 12.0),
            (4, 5): (60.0, 25.0),
            (5, 4): (60.0, 25.0),
            (5, 6): (40.0, 18.0),
            (6, 5): (40.0, 18.0),
        }

    def get_transition_prior(self, cam_src: int, cam_dst: int) -> float:
        """Adjacency transition probability."""
        if cam_src == cam_dst:
            return 0.8
        src_k = f"c{cam_src}"
        dst_k = f"c{cam_dst}"
        if src_k in self.transition_matrix and dst_k in self.transition_matrix[src_k]:
            return float(self.transition_matrix[src_k][dst_k])
        # Fallback based on camera index distance
        cam_dist = abs(cam_src - cam_dst)
        return float(max(0.05, 1.0 / (1.0 + cam_dist)))

    def compute_spatiotemporal_likelihood(self, cam_src: int, t_src: float,
                                           cam_dst: int, t_dst: float) -> float:
        """
        Calculates P(Delta t | Cam_src -> Cam_dst) * P(Cam_src -> Cam_dst).
        """
        dt = t_dst - t_src
        if dt < 0: # Time inversion
            return 0.0

        prior = self.get_transition_prior(cam_src, cam_dst)

        if cam_src == cam_dst:
            # Same camera: smoothly decays over hours
            temporal_decay = np.exp(-dt / 1800.0)
            return float(prior * temporal_decay)

        pair = (cam_src, cam_dst)
        if pair in self.travel_time_params:
            mu, sigma = self.travel_time_params[pair]
            # Gaussian likelihood
            prob_dt = np.exp(-0.5 * ((dt - mu) / (sigma + 1e-5)) ** 2)
        else:
            # Uncalibrated pair: smooth log-normal / exponential window
            expected_time = abs(cam_src - cam_dst) * 40.0
            prob_dt = np.exp(-0.5 * ((dt - expected_time) / 60.0) ** 2)

        return float(np.clip(prior * prob_dt, 0.0, 1.0))
