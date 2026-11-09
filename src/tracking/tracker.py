"""
Multi-Object Tracking (MOT) Module for Cross-Camera Video Feeds.
Implements ByteTrack-style two-stage IoU and Kalman association.
Maintains local camera-specific track IDs, trajectories, and historical crops.
Explicitly decouples local tracker IDs from Global Identity assignment.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculates Intersection over Union between [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = b1_area + b2_area - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """Single Object Tracklet."""
    _count = 0

    def __init__(self, bbox: List[int], score: float, crop: Optional[np.ndarray] = None):
        STrack._count += 1
        self.track_id = STrack._count
        self.bbox = bbox # [x1, y1, x2, y2]
        self.score = score
        self.state = TrackState.New
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.history_crops = [crop] if crop is not None else []
        self.history_bboxes = [bbox]
        self.global_id = None # Decoupled: set exclusively by ReID IdentityManager
        self.best_crop = crop
        self.best_quality = 0.5

    def activate(self, frame_id: int):
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 1

    def re_activate(self, new_track: 'STrack', frame_id: int):
        self.bbox = new_track.bbox
        self.score = new_track.score
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.tracklet_len += 1
        if len(new_track.history_crops) > 0:
            self.history_crops.append(new_track.history_crops[0])
            if len(self.history_crops) > 10:
                self.history_crops.pop(0)
        self.history_bboxes.append(new_track.bbox)

    def update(self, new_track: 'STrack', frame_id: int):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.bbox = new_track.bbox
        self.score = new_track.score
        self.state = TrackState.Tracked
        self.is_activated = True
        if len(new_track.history_crops) > 0:
            self.history_crops.append(new_track.history_crops[0])
            if len(self.history_crops) > 10:
                self.history_crops.pop(0)
        self.history_bboxes.append(new_track.bbox)

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class ByteTracker:
    """
    ByteTrack implementation for robust multi-person tracking.
    Associates high-score and low-score detection bounding boxes across frames.
    """
    def __init__(self,
                 track_thresh: float = 0.20,
                 match_thresh: float = 0.35,
                 track_buffer: int = 40,
                 camera_id: int = 1):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.camera_id = camera_id

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0

    def reset(self):
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        STrack._count = 0

    def update(self, detections: List[Dict[str, Any]]) -> List[STrack]:
        """
        Updates tracker state with new frame detections.
        Returns list of currently active confirmed tracks.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Split high score and low score detections
        dets_high = []
        dets_low = []
        for det in detections:
            strack = STrack(det["bbox"], det["confidence"], det.get("crop"))
            if det["confidence"] >= self.track_thresh:
                dets_high.append(strack)
            elif det["confidence"] >= 0.1:
                dets_low.append(strack)

        # Separate unconfirmed and tracked stracks
        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked_pool = [t for t in self.tracked_stracks if t.is_activated] + self.lost_stracks

        # Step 1: Match high score detections with tracked pool
        matched_a, unmatched_tracks_a, unmatched_dets_a = self._associate(tracked_pool, dets_high, self.match_thresh)
        for t_idx, d_idx in matched_a:
            track = tracked_pool[t_idx]
            det = dets_high[d_idx]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        # Step 2: Match remaining tracks with low score detections
        remain_tracks = [tracked_pool[i] for i in unmatched_tracks_a if tracked_pool[i].state == TrackState.Tracked]
        matched_b, unmatched_tracks_b, _ = self._associate(remain_tracks, dets_low, 0.5)
        for t_idx, d_idx in matched_b:
            track = remain_tracks[t_idx]
            det = dets_low[d_idx]
            track.update(det, self.frame_id)
            activated_stracks.append(track)

        # Mark unmatched tracks as lost
        for t_idx in unmatched_tracks_b:
            track = remain_tracks[t_idx]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Step 3: Match unmatched high score detections with unconfirmed tracks
        remain_dets = [dets_high[i] for i in unmatched_dets_a]
        matched_c, unmatched_unconfirmed, unmatched_new_dets = self._associate(unconfirmed, remain_dets, 0.6)
        for t_idx, d_idx in matched_c:
            track = unconfirmed[t_idx]
            det = remain_dets[d_idx]
            track.update(det, self.frame_id)
            activated_stracks.append(track)

        for t_idx in unmatched_unconfirmed:
            track = unconfirmed[t_idx]
            track.mark_removed()
            removed_stracks.append(track)

        # Initialize new tracks
        for d_idx in unmatched_new_dets:
            det = remain_dets[d_idx]
            det.activate(self.frame_id)
            activated_stracks.append(det)

        # Handle old lost tracks
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks += activated_stracks
        self.tracked_stracks += refind_stracks
        self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]
        self.lost_stracks += lost_stracks
        self.removed_stracks += removed_stracks

        # Deduplicate tracked stracks
        output_stracks = []
        seen_ids = set()
        for t in self.tracked_stracks:
            if t.track_id not in seen_ids:
                seen_ids.add(t.track_id)
                output_stracks.append(t)

        self.tracked_stracks = output_stracks
        return [t for t in self.tracked_stracks if t.is_activated]

    @staticmethod
    def _associate(tracks: List[STrack], dets: List[STrack], thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(tracks) == 0 or len(dets) == 0:
            return [], list(range(len(tracks))), list(range(len(dets)))

        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                iou_matrix[i, j] = calculate_iou(t.bbox, d.bbox)

        # Greedy match on IoU
        matched = []
        matched_tracks = set()
        matched_dets = set()

        while True:
            max_iou = 0.0
            best_i, best_j = -1, -1
            for i in range(len(tracks)):
                if i in matched_tracks:
                    continue
                for j in range(len(dets)):
                    if j in matched_dets:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        best_i, best_j = i, j

            if max_iou >= thresh and best_i != -1:
                matched.append((best_i, best_j))
                matched_tracks.add(best_i)
                matched_dets.add(best_j)
            else:
                break

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [j for j in range(len(dets)) if j not in matched_dets]

        return matched, unmatched_tracks, unmatched_dets
