#!/usr/bin/env python3
"""
M²-ReID Master Execution Pipeline.
Unified CLI to orchestrate datasets, model training, evaluation, video inference,
ablation benchmarks, and the Streamlit research dashboard.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import yaml
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def parse_args():
    parser = argparse.ArgumentParser(description="M²-ReID: Adaptive Multi-Model Cross-Camera Person Re-Identification")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["check", "demo_data", "experiments", "ablation", "robustness", "msmt17", "video", "error_analysis", "dashboard", "all"],
                        help="Execution mode")
    parser.add_argument("--video_path", type=str, default=None,
                        help="Path to input video file (e.g. 'my_video.mp4') or camera index (e.g. '0' for live webcam)")
    parser.add_argument("--output_video", type=str, default="results/annotated_output.mp4",
                        help="Path to save annotated output video with bounding boxes, global IDs, and tracks")
    parser.add_argument("--output_json", type=str, default="results/video_reid_results.json",
                        help="Path to save detailed JSON detection & Re-ID tracking results")
    parser.add_argument("--camera_id", type=int, default=1, help="Camera ID for video stream")
    parser.add_argument("--conf_thresh", type=float, default=0.45, help="YOLO person detection confidence threshold")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    return parser.parse_args()

def _get_id_color(identity_str: str) -> tuple:
    """Generates consistent RGB color for an identity string."""
    hash_val = hash(identity_str)
    r = (hash_val & 0xFF) % 200 + 55
    g = ((hash_val >> 8) & 0xFF) % 200 + 55
    b = ((hash_val >> 16) & 0xFF) % 200 + 55
    return (int(b), int(g), int(r)) # BGR for OpenCV

def run_video_pipeline(video_path: str,
                       camera_id: int = 1,
                       output_video: str = None,
                       output_json: str = None,
                       conf_thresh: float = 0.45,
                       device: str = "cpu"):
    """
    Executes full video detection, tracking, feature extraction, and global identity association.
    Optionally renders annotated frames to an output video and saves structured tracking JSON.
    """
    import json
    import cv2
    from src.detection.yolo_detector import PersonDetector
    from src.tracking.tracker import ByteTracker
    from src.reid.model_factory import build_reid_model
    from src.fusion.adaptive_fusion import AdaptiveMultiModelFusion
    from src.identity.identity_manager import GlobalIdentityManager
    from src.quality.quality_estimator import ImageQualityEstimator
    from src.pose.pose_estimator import PoseEstimator
    from src.data.transforms import build_reid_transforms
    from PIL import Image

    print(f"\n[VIDEO PIPELINE] Processing video: {video_path} (Camera {camera_id}) on {device.upper()}")
    
    # Handle webcam integer index or file path
    video_source = int(video_path) if str(video_path).isdigit() else video_path
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps: # Check for NaN or 0
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer if requested
    writer = None
    if output_video:
        out_path = Path(output_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        print(f"[VIDEO PIPELINE] Saving annotated video to: {output_video} ({width}x{height} @ {fps:.1f} FPS)")

    detector = PersonDetector(conf_thresh=conf_thresh, device=device)
    print(f"[VIDEO PIPELINE] Person detector backend: {detector.backend.upper()}")
    tracker = ByteTracker(camera_id=camera_id)
    id_manager = GlobalIdentityManager()
    quality_estimator = ImageQualityEstimator()
    pose_estimator = PoseEstimator().to(device)
    test_transform = build_reid_transforms(256, 128, is_train=False)

    model_osnet = build_reid_model("osnet", device=device).eval()
    model_trans = build_reid_model("transreid", device=device).eval()
    model_clip = build_reid_model("clip", device=device).eval()
    adaptive_model = AdaptiveMultiModelFusion().to(device).eval()

    frame_idx = 0
    total_detections = 0
    frame_records = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        timestamp = frame_idx / float(fps)

        # 1. Detection
        detections = detector.detect(frame)
        total_detections += len(detections)

        # 2. Tracking
        active_tracks = tracker.update(detections)

        # 3. Feature Extraction & Global Identity Association
        for tr in active_tracks:
            if tr.best_crop is not None and tr.best_crop.size > 0:
                crop_pil = Image.fromarray(cv2.cvtColor(tr.best_crop, cv2.COLOR_BGR2RGB))
                crop_tensor = test_transform(crop_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    f_os = model_osnet(crop_tensor)
                    f_tr = model_trans(crop_tensor)
                    f_cl = model_clip(crop_tensor)
                    fused_emb, weights = adaptive_model(f_os, f_tr, f_cl)

                emb_np = fused_emb.squeeze(0).cpu().numpy()
                q_score = quality_estimator.extract_quality_vector(tr.best_crop)[0]
                p_vec = pose_estimator.extract_pose_feature(tr.best_crop)

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

        # 4. Annotate frame and record tracking data
        current_tracked = []
        for tr in active_tracks:
            gid = tr.global_id or f"Trk-{tr.track_id}"
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            color = _get_id_color(gid)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label badge
            label = f"{gid} | Trk #{tr.track_id} ({tr.score:.2f})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + lw + 6, max(22, y1)), color, -1)
            cv2.putText(frame, label, (x1 + 3, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            current_tracked.append({
                "global_id": gid,
                "track_id": tr.track_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(float(tr.score), 3)
            })

        # Draw HUD overlay
        hud = f"M2-ReID | Frame: {frame_idx} | Active: {len(active_tracks)} | Unique IDs: {len(id_manager.identities)}"
        cv2.rectangle(frame, (10, 10), (550, 40), (20, 20, 20), -1)
        cv2.putText(frame, hud, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)

        if writer:
            writer.write(frame)

        frame_records.append({
            "frame_idx": frame_idx,
            "timestamp": round(timestamp, 2),
            "tracked_count": len(current_tracked),
            "persons": current_tracked
        })

    cap.release()
    if writer:
        writer.release()
        print(f"[VIDEO PIPELINE] Annotated video saved successfully to: {output_video}")

    # Save JSON results
    if output_json:
        out_json_path = Path(output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip non-JSON-serializable fields (numpy arrays like best_crop)
        clean_summaries = []
        for s in id_manager.get_identity_summary():
            clean_summaries.append({
                k: v for k, v in s.items()
                if not isinstance(v, np.ndarray)
            })
        results_payload = {
            "video_path": str(video_path),
            "camera_id": camera_id,
            "total_frames_processed": frame_idx,
            "total_detections": total_detections,
            "unique_identities_count": len(id_manager.identities),
            "identity_summaries": clean_summaries,
            "frame_records": frame_records
        }
        with open(out_json_path, "w") as f:
            json.dump(results_payload, f, indent=2)
        print(f"[VIDEO PIPELINE] Detection & Re-ID results saved to: {output_json}")

    print(f"[VIDEO PIPELINE] Processed {frame_idx} frames. Total Detections: {total_detections}")
    print(f"[VIDEO PIPELINE] Registered {len(id_manager.identities)} unique Global Person Identities.")
    for s in id_manager.get_identity_summary():
        print(f"  -> Global ID: {s['global_id']} | Sightings: {s['num_sightings']} | Visited: Cam {s['cameras_visited']} | Path: {s['timeline_str']}")

def main():
    args = parse_args()
    device = "cuda" if (torch.cuda.is_available() and args.device in ["auto", "cuda"]) else "cpu"

    print("\n" + "#" * 75)
    print("      M²-ReID: Adaptive Multi-Model Person Re-Identification")
    print("      Research Framework & Benchmark Suite")
    print("#" * 75 + "\n")

    if args.mode in ["check", "all"]:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "check_datasets.py")])

    if args.mode in ["demo_data", "all"]:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "generate_demo_data.py")])

    if args.mode in ["experiments", "all"]:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "experiments" / "run_experiments.py")])

    if args.mode in ["ablation", "all"]:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "experiments" / "run_ablation.py")])

    if args.mode in ["robustness", "all"]:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "experiments" / "run_robustness.py")])

    if args.mode in ["msmt17", "all"]:
        subprocess.run([sys.executable, str(PROJECT_ROOT / "experiments" / "run_msmt17_eval.py")])

    if args.mode in ["video", "all"]:
        if args.video_path:
            video_input = args.video_path
        else:
            demo_vid = PROJECT_ROOT / "data" / "demo_videos" / "cam1_demo.mp4"
            if not demo_vid.exists():
                from scripts.generate_demo_data import generate_demo_video
                generate_demo_video(PROJECT_ROOT)
            video_input = str(demo_vid)

        run_video_pipeline(
            video_path=video_input,
            camera_id=args.camera_id,
            output_video=args.output_video,
            output_json=args.output_json,
            conf_thresh=args.conf_thresh,
            device=device
        )

    if args.mode == "dashboard":
        print("[DASHBOARD] Launching Streamlit Dashboard...")
        subprocess.run(["streamlit", "run", str(PROJECT_ROOT / "dashboard" / "app.py")])

    print("\n[M²-ReID] Pipeline execution finished successfully!")

if __name__ == "__main__":
    main()
