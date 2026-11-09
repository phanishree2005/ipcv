"""
Integration tests for End-to-End M²-ReID modules:
- DatasetValidator
- Quality Estimator
- Pose Estimator
- ByteTracker
- GlobalIdentityManager
"""

import numpy as np
import cv2
import pytest
from src.data.dataset_validator import DatasetValidator
from src.quality.quality_estimator import ImageQualityEstimator
from src.pose.pose_estimator import PoseEstimator
from src.tracking.tracker import ByteTracker
from src.identity.identity_manager import GlobalIdentityManager

def test_quality_estimator():
    estimator = ImageQualityEstimator()
    dummy_img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    q_vec = estimator.extract_quality_vector(dummy_img, detection_conf=0.92)
    assert q_vec.shape == (7,)
    assert 0.0 <= q_vec[0] <= 1.0 # blur
    assert 0.0 <= q_vec[1] <= 1.0 # brightness

def test_pose_estimator():
    estimator = PoseEstimator()
    dummy_img = np.zeros((256, 128, 3), dtype=np.uint8)
    p_vec = estimator.extract_pose_feature(dummy_img)
    assert p_vec.shape == (34,)
    sim = estimator.compute_pose_similarity(p_vec, p_vec)
    assert 0.9 <= sim <= 1.0

def test_tracker_and_identity_manager():
    tracker = ByteTracker(camera_id=1)
    id_mgr = GlobalIdentityManager()

    # Step 1: Detect and track 2 persons in Frame 1
    dummy_crop1 = np.ones((100, 50, 3), dtype=np.uint8) * 50
    dummy_crop2 = np.ones((100, 50, 3), dtype=np.uint8) * 150

    dets = [
        {"bbox": [20, 20, 70, 120], "confidence": 0.90, "crop": dummy_crop1},
        {"bbox": [200, 20, 250, 120], "confidence": 0.85, "crop": dummy_crop2},
    ]
    tracks = tracker.update(dets)
    assert len(tracks) == 2

    # Step 2: Associate with Global Identity Manager
    emb1 = np.random.randn(512).astype(np.float32)
    emb2 = np.random.randn(512).astype(np.float32)

    gid1, _, _, _ = id_mgr.get_or_create_identity(emb1, camera_id=1, timestamp=10.0, track_id=tracks[0].track_id, crop=dummy_crop1)
    gid2, _, _, _ = id_mgr.get_or_create_identity(emb2, camera_id=1, timestamp=10.0, track_id=tracks[1].track_id, crop=dummy_crop2)

    assert gid1 == "P001"
    assert gid2 == "P002"
    assert len(id_mgr.identities) == 2
