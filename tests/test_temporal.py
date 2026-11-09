import pytest
import numpy as np
from src.temporal.camera_topology import CameraTopology
from src.temporal.temporal_reasoning import TemporalReasoningEngine

def test_camera_topology_transition():
    topology = CameraTopology(num_cameras=6)
    # Transition prior between camera 1 and 2
    prior = topology.get_transition_prior(1, 2)
    assert 0.0 <= prior <= 1.0

    # Spatiotemporal likelihood
    likelihood = topology.compute_spatiotemporal_likelihood(cam_src=1, t_src=10.0, cam_dst=2, t_dst=50.0)
    assert 0.0 <= likelihood <= 1.0

def test_temporal_reasoning_engine():
    engine = TemporalReasoningEngine(visual_weight=0.7)
    score, breakdown = engine.evaluate_match(
        visual_similarity=0.85,
        cam_query=2,
        time_query=60.0,
        cam_candidate=1,
        time_candidate=10.0
    )
    assert 0.0 <= score <= 1.0
    assert "visual_similarity" in breakdown
    assert "spatiotemporal_likelihood" in breakdown
    assert "combined_score" in breakdown
    assert breakdown["visual_similarity"] == 0.85
