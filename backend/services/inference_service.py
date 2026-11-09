import os
import cv2
import time
import base64
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Any, Optional

from src.detection.yolo_detector import PersonDetector
from src.tracking.tracker import ByteTracker
from src.identity.identity_manager import GlobalIdentityManager
from src.quality.quality_estimator import ImageQualityEstimator
from src.pose.pose_estimator import PoseEstimator
from src.reid.model_factory import build_reid_model
from src.fusion.adaptive_fusion import AdaptiveMultiModelFusion
from src.data.transforms import build_reid_transforms

def _get_id_color(gid: str) -> tuple:
    if not gid:
        return (0, 255, 0)
    hash_val = sum(ord(c) for c in gid)
    colors = [
        (255, 99, 71),   # Tomato
        (30, 144, 255),  # DodgerBlue
        (50, 205, 50),   # LimeGreen
        (255, 215, 0),   # Gold
        (186, 85, 211),  # MediumOrchid
        (0, 206, 209),   # DarkTurquoise
        (255, 140, 0),   # DarkOrange
        (238, 130, 238), # Violet
    ]
    return colors[hash_val % len(colors)]

def _frame_to_base64(img_bgr: np.ndarray, quality: int = 80) -> str:
    _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode('utf-8')

class FullPipelineInferenceService:
    def __init__(self, device: str = "cpu", conf_thresh: float = 0.25):
        self.device = device
        self.conf_thresh = conf_thresh
        print(f"[InferenceService] Initializing models on {device}...")
        
        self.detector = PersonDetector(conf_thresh=conf_thresh, device=device)
        self.quality_estimator = ImageQualityEstimator()
        self.pose_estimator = PoseEstimator().to(device)
        self.test_transform = build_reid_transforms(256, 128, is_train=False)

        self.model_osnet = build_reid_model("osnet", device=device).eval()
        self.model_trans = build_reid_model("transreid", device=device).eval()
        self.model_clip = build_reid_model("clip", device=device).eval()
        self.adaptive_model = AdaptiveMultiModelFusion().to(device).eval()
        
        # State for live webcam session
        self.live_tracker = ByteTracker(camera_id=0)
        self.live_id_manager = GlobalIdentityManager()
        self.live_frame_idx = 0
        
        # Persistent identity manager across multi-camera video uploads
        self.video_id_manager = GlobalIdentityManager()
        print("[InferenceService] All pipeline components ready.")

    def reset_live_session(self):
        self.live_tracker = ByteTracker(camera_id=0)
        self.live_id_manager = GlobalIdentityManager()
        self.live_frame_idx = 0

    def process_single_frame(self, frame_bgr: np.ndarray, camera_id: int = 0) -> Dict[str, Any]:
        """
        Processes a single live webcam frame, performs YOLOv8 detection,
        ByteTrack tracking, and Multi-Model Re-ID identification.
        """
        self.live_frame_idx += 1
        t_start = time.time()
        
        # 1. Detection
        detections = self.detector.detect(frame_bgr)
        
        # 2. Tracking
        active_tracks = self.live_tracker.update(detections)
        
        # 3. Feature Extraction & Global ID
        fused_weights_summary = [0.33, 0.33, 0.34]
        for tr in active_tracks:
            if tr.best_crop is not None and tr.best_crop.size > 0:
                crop_pil = Image.fromarray(cv2.cvtColor(tr.best_crop, cv2.COLOR_BGR2RGB))
                crop_tensor = self.test_transform(crop_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    f_os = self.model_osnet(crop_tensor)
                    f_tr = self.model_trans(crop_tensor)
                    f_cl = self.model_clip(crop_tensor)
                    fused_emb, weights = self.adaptive_model(f_os, f_tr, f_cl)
                    fused_weights_summary = [float(w) for w in weights.squeeze(0).cpu().numpy()]

                emb_np = fused_emb.squeeze(0).cpu().numpy()
                q_score = self.quality_estimator.extract_quality_vector(tr.best_crop)[0]
                p_vec = self.pose_estimator.extract_pose_feature(tr.best_crop)

                gid, sim, conf, _ = self.live_id_manager.get_or_create_identity(
                    embedding=emb_np,
                    camera_id=camera_id,
                    timestamp=self.live_frame_idx * 0.033,
                    track_id=tr.track_id,
                    quality_score=q_score,
                    pose_vector=p_vec,
                    crop=tr.best_crop,
                    bbox=tr.bbox
                )
                tr.global_id = gid

        # 4. Annotate
        annotated_frame = frame_bgr.copy()
        current_tracked = []
        for tr in active_tracks:
            gid = tr.global_id or f"Trk-{tr.track_id}"
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            color = _get_id_color(gid)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{gid} | Trk #{tr.track_id} ({tr.score:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), color, -1)
            cv2.putText(annotated_frame, label, (x1 + 3, max(12, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            current_tracked.append({
                "global_id": gid,
                "track_id": tr.track_id,
                "confidence": float(tr.score),
                "bbox": [x1, y1, x2, y2]
            })

        # Top HUD
        fps = 1.0 / max(1e-5, (time.time() - t_start))
        hud = f"M2-ReID Live | Frame: {self.live_frame_idx} | Active: {len(active_tracks)} | Unique IDs: {len(self.live_id_manager.identities)} | FPS: {fps:.1f}"
        cv2.rectangle(annotated_frame, (0, 0), (640, 28), (20, 20, 20), -1)
        cv2.putText(annotated_frame, hud, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

        # Build clean summaries
        id_summaries = []
        for s in self.live_id_manager.get_identity_summary():
            crop_b64 = None
            if s.get("best_crop") is not None and isinstance(s["best_crop"], np.ndarray):
                crop_b64 = _frame_to_base64(s["best_crop"], quality=85)
            id_summaries.append({
                "global_id": str(s.get("global_id", "P???")),
                "sighting_count": int(s.get("num_sightings", 1)),
                "cameras_seen": list(s.get("cameras_visited", [camera_id])),
                "first_seen": str(s.get("first_seen", "0s")),
                "last_seen": str(s.get("last_seen", "0s")),
                "confidence": float(s.get("confidence", 0.88)),
                "crop_base64": crop_b64
            })

        return {
            "frame_id": self.live_frame_idx,
            "fps": round(fps, 1),
            "annotated_frame_base64": _frame_to_base64(annotated_frame, quality=80),
            "detections_count": len(detections),
            "active_tracks": current_tracked,
            "identities": id_summaries,
            "weights": {
                "osnet": round(fused_weights_summary[0], 3),
                "transreid": round(fused_weights_summary[1], 3),
                "clip": round(fused_weights_summary[2], 3),
            }
        }

    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        max_frames: int = 300,
        camera_id: int = 1
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline on a video file and generates
        annotated video + comprehensive JSON data.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Downscale 4K/UHD videos to 720p HD (max width 1280) for real-time CPU speed
        scale = 1.0
        out_w, out_h = width, height
        if width > 1280:
            scale = 1280.0 / width
            out_w = 1280
            out_h = int(height * scale)

        writer = None
        if output_video_path:
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))

        tracker = ByteTracker(camera_id=camera_id)
        id_manager = self.video_id_manager

        frame_idx = 0
        total_detections = 0
        frame_records = []
        keyframe_snapshots = []

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames > 0 and frame_idx >= max_frames):
                break
            frame_idx += 1
            timestamp = frame_idx / float(fps)

            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h))

            detections = self.detector.detect(frame)
            total_detections += len(detections)

            active_tracks = tracker.update(detections)

            for tr in active_tracks:
                if tr.best_crop is not None and tr.best_crop.size > 0:
                    crop_pil = Image.fromarray(cv2.cvtColor(tr.best_crop, cv2.COLOR_BGR2RGB))
                    crop_tensor = self.test_transform(crop_pil).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        f_os = self.model_osnet(crop_tensor)
                        f_tr = self.model_trans(crop_tensor)
                        f_cl = self.model_clip(crop_tensor)
                        fused_emb, weights = self.adaptive_model(f_os, f_tr, f_cl)

                    emb_np = fused_emb.squeeze(0).cpu().numpy()
                    q_score = self.quality_estimator.extract_quality_vector(tr.best_crop)[0]
                    p_vec = self.pose_estimator.extract_pose_feature(tr.best_crop)

                    gid, sim, conf, _ = id_manager.get_or_create_identity(
                        embedding=emb_np,
                        camera_id=camera_id,
                        timestamp=timestamp,
                        track_id=tr.track_id,
                        quality_score=q_score,
                        pose_vector=p_vec,
                        crop=tr.best_crop,
                        bbox=tr.bbox
                    )
                    tr.global_id = gid

            annotated_frame = frame.copy()
            current_tracked = []
            for tr in active_tracks:
                gid = tr.global_id or f"Trk-{tr.track_id}"
                x1, y1, x2, y2 = [int(v) for v in tr.bbox]
                color = _get_id_color(gid)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{gid} | Trk #{tr.track_id} ({tr.score:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), color, -1)
                cv2.putText(annotated_frame, label, (x1 + 3, max(12, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                current_tracked.append({
                    "track_id": tr.track_id,
                    "global_id": gid,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(float(tr.score), 3)
                })

            hud = f"M2-ReID | Frame: {frame_idx}/{total_video_frames} | Active: {len(active_tracks)} | Unique IDs: {len(id_manager.identities)}"
            cv2.rectangle(annotated_frame, (10, 10), (550, 40), (20, 20, 20), -1)
            cv2.putText(annotated_frame, hud, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)

            if writer is not None:
                writer.write(annotated_frame)

            # Store keyframes (e.g. every 25 frames or when detections exist)
            if frame_idx % 20 == 1 or (len(current_tracked) > 0 and len(keyframe_snapshots) < 8):
                keyframe_snapshots.append({
                    "frame_idx": frame_idx,
                    "timestamp": round(timestamp, 2),
                    "snapshot_b64": _frame_to_base64(annotated_frame, quality=70)
                })

            frame_records.append({
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 2),
                "num_detections": len(detections),
                "tracked_objects": current_tracked
            })

        cap.release()
        if writer is not None:
            writer.release()

        # Build clean identity summaries with base64 crops
        clean_summaries = []
        for s in id_manager.get_identity_summary():
            crop_b64 = None
            if s.get("best_crop") is not None and isinstance(s["best_crop"], np.ndarray):
                crop_b64 = _frame_to_base64(s["best_crop"], quality=85)

            gid_str = str(s.get("global_id", "P???"))
            emb = None
            if gid_str in id_manager.identities:
                emb = id_manager.identities[gid_str].representative_embedding

            clean_summaries.append({
                "global_id": gid_str,
                "sighting_count": int(s.get("num_sightings", 1)),
                "cameras_seen": list(s.get("cameras_visited", [camera_id])),
                "first_seen": str(s.get("first_seen", "0s")),
                "last_seen": str(s.get("last_seen", "0s")),
                "confidence": float(s.get("confidence", 0.88)),
                "crop_base64": crop_b64,
                "embedding": emb
            })

        return {
            "video_path": video_path,
            "output_video_path": output_video_path,
            "total_frames_processed": frame_idx,
            "total_video_frames": total_video_frames,
            "fps": fps,
            "resolution": [width, height],
            "total_detections": total_detections,
            "unique_identities_count": len(clean_summaries),
            "identity_summaries": clean_summaries,
            "keyframes": keyframe_snapshots,
            "frame_sample": frame_records[:50]  # sample of frame records
        }
