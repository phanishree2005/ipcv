#!/usr/bin/env python3
"""
M²-ReID Demo Data Generator
Generates realistic sample person images and multi-camera simulated sequence crops
for Market-1501 and MSMT17 structure validation and immediate end-to-end testing.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def generate_person_crop(pid: int, camid: int, frame_idx: int, width: int = 128, height: int = 256) -> np.ndarray:
    """
    Generates a deterministic person silhouette with distinct identity colors,
    clothing pattern, illumination variation per camera, and realistic textures.
    """
    np.random.seed(pid * 1000 + 42)
    # Identity appearance attributes
    torso_color = (
        int(np.random.randint(40, 240)),
        int(np.random.randint(40, 240)),
        int(np.random.randint(40, 240))
    )
    pants_color = (
        int(np.random.randint(20, 220)),
        int(np.random.randint(20, 220)),
        int(np.random.randint(20, 220))
    )
    skin_tone = (
        int(np.random.randint(180, 240)),
        int(np.random.randint(140, 190)),
        int(np.random.randint(100, 150))
    )
    hair_color = (
        int(np.random.randint(20, 70)),
        int(np.random.randint(15, 60)),
        int(np.random.randint(10, 50))
    )

    # Camera illumination factor
    cam_illumination = {
        1: 1.0, 2: 0.85, 3: 1.15, 4: 0.90, 5: 1.05, 6: 0.75
    }.get(camid, 1.0)

    # Create background
    img = np.full((height, width, 3), (120 + (camid * 15) % 80, 125, 130), dtype=np.uint8)
    # Add subtle background texture / noise
    noise = np.random.normal(0, 10, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    center_x = width // 2 + int(np.sin(frame_idx * 0.1) * 3)
    
    # Head
    head_cy = int(height * 0.16)
    head_rx = int(width * 0.14)
    head_ry = int(height * 0.08)
    cv2.ellipse(img, (center_x, head_cy), (head_rx, head_ry), 0, 0, 360, skin_tone, -1)
    # Hair
    cv2.ellipse(img, (center_x, head_cy - int(head_ry * 0.4)), (head_rx, int(head_ry * 0.6)), 0, 180, 360, hair_color, -1)

    # Torso (Upper body)
    torso_top = int(height * 0.25)
    torso_bottom = int(height * 0.60)
    torso_w = int(width * 0.32)
    pt1 = (center_x - torso_w, torso_top)
    pt2 = (center_x + torso_w, torso_bottom)
    cv2.rectangle(img, pt1, pt2, torso_color, -1)
    
    # Arms
    arm_w = int(width * 0.08)
    cv2.rectangle(img, (center_x - torso_w - arm_w, torso_top), (center_x - torso_w, torso_bottom - 10), torso_color, -1)
    cv2.rectangle(img, (center_x + torso_w, torso_top), (center_x + torso_w + arm_w, torso_bottom - 10), torso_color, -1)

    # Legs (Lower body)
    legs_bottom = int(height * 0.92)
    leg_w = int(width * 0.13)
    leg_gap = int(width * 0.04)
    # Left leg
    cv2.rectangle(img, (center_x - leg_gap - leg_w, torso_bottom), (center_x - leg_gap, legs_bottom), pants_color, -1)
    # Right leg
    cv2.rectangle(img, (center_x + leg_gap, torso_bottom), (center_x + leg_gap + leg_w, legs_bottom), pants_color, -1)

    # Shoes
    shoe_color = (30, 30, 30)
    cv2.rectangle(img, (center_x - leg_gap - leg_w - 2, legs_bottom), (center_x - leg_gap, height - 8), shoe_color, -1)
    cv2.rectangle(img, (center_x + leg_gap, legs_bottom), (center_x + leg_gap + leg_w + 2, height - 8), shoe_color, -1)

    # Apply camera illumination
    img = np.clip(img.astype(np.float32) * cam_illumination, 0, 255).astype(np.uint8)
    return img

def generate_market1501_demo(base_dir: Path, num_train_ids=20, num_test_ids=15, num_cams=6):
    print("Generating Market-1501 sample data...")
    market_root = base_dir / "data" / "market1501" / "Market-1501-v15.09.15"
    train_dir = market_root / "bounding_box_train"
    query_dir = market_root / "query"
    gallery_dir = market_root / "bounding_box_test"
    gt_dir = market_root / "gt_bbox"

    for d in [train_dir, query_dir, gallery_dir, gt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Generate Train Split (IDs 1 to num_train_ids)
    for pid in range(1, num_train_ids + 1):
        for cam in range(1, num_cams + 1):
            for seq in range(1, 3):
                for f_idx in range(1, 3):
                    fn = f"{pid:04d}_c{cam}s{seq}_{f_idx:06d}_00.jpg"
                    img = generate_person_crop(pid, cam, f_idx)
                    cv2.imwrite(str(train_dir / fn), img)

    # Generate Query Split (IDs from num_train_ids + 1 to num_train_ids + num_test_ids)
    test_id_start = num_train_ids + 1
    test_id_end = num_train_ids + num_test_ids
    for pid in range(test_id_start, test_id_end + 1):
        # Query: 1 or 2 viewpoints
        for cam in [1, 2]:
            fn = f"{pid:04d}_c{cam}s1_000001_00.jpg"
            img = generate_person_crop(pid, cam, 1)
            cv2.imwrite(str(query_dir / fn), img)

    # Generate Gallery Split (IDs from test split across multiple other cameras)
    for pid in range(test_id_start, test_id_end + 1):
        for cam in range(1, num_cams + 1):
            for seq in range(1, 3):
                for f_idx in range(1, 3):
                    fn = f"{pid:04d}_c{cam}s{seq}_{f_idx:06d}_00.jpg"
                    img = generate_person_crop(pid, cam, f_idx)
                    cv2.imwrite(str(gallery_dir / fn), img)

    # Add a few distractors / junk (PID = 0, PID = -1)
    for distractor_id in [0, -1]:
        for idx in range(1, 6):
            fn = f"{distractor_id:04d}_c1s1_{idx:06d}_00.jpg"
            img = generate_person_crop(999 + idx, 1, idx)
            cv2.imwrite(str(gallery_dir / fn), img)

    print(f"Market-1501 sample data generated successfully at: {market_root}")

def generate_msmt17_demo(base_dir: Path, num_train_ids=20, num_test_ids=15, num_cams=8):
    print("Generating MSMT17 sample data...")
    msmt_root = base_dir / "data" / "msmt17" / "MSMT17_V1"
    train_dir = msmt_root / "train"
    test_dir = msmt_root / "test"

    for d in [train_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    list_train = msmt_root / "list_train.txt"
    list_val = msmt_root / "list_val.txt"
    list_query = msmt_root / "list_query.txt"
    list_gallery = msmt_root / "list_gallery.txt"

    train_lines = []
    val_lines = []
    query_lines = []
    gallery_lines = []

    # Train identities
    for pid in range(1, num_train_ids + 1):
        pid_dir = train_dir / f"{pid:04d}"
        pid_dir.mkdir(parents=True, exist_ok=True)
        for cam in range(1, num_cams + 1):
            for f_idx in range(1, 3):
                fn = f"{pid:04d}_c{cam:02d}_{f_idx:04d}.jpg"
                img = generate_person_crop(pid, cam, f_idx)
                cv2.imwrite(str(pid_dir / fn), img)
                rel_path = f"{pid:04d}/{fn}"
                if f_idx == 1:
                    train_lines.append(f"{rel_path} {pid}")
                else:
                    val_lines.append(f"{rel_path} {pid}")

    # Test identities
    test_start = num_train_ids + 1
    test_end = num_train_ids + num_test_ids
    for pid in range(test_start, test_end + 1):
        pid_dir = test_dir / f"{pid:04d}"
        pid_dir.mkdir(parents=True, exist_ok=True)
        for cam in range(1, num_cams + 1):
            for f_idx in range(1, 3):
                fn = f"{pid:04d}_c{cam:02d}_{f_idx:04d}.jpg"
                img = generate_person_crop(pid, cam, f_idx)
                cv2.imwrite(str(pid_dir / fn), img)
                rel_path = f"{pid:04d}/{fn}"
                if cam in [1, 2] and f_idx == 1:
                    query_lines.append(f"{rel_path} {pid}")
                else:
                    gallery_lines.append(f"{rel_path} {pid}")

    with open(list_train, "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(list_val, "w") as f:
        f.write("\n".join(val_lines) + "\n")
    with open(list_query, "w") as f:
        f.write("\n".join(query_lines) + "\n")
    with open(list_gallery, "w") as f:
        f.write("\n".join(gallery_lines) + "\n")

    print(f"MSMT17 sample data generated successfully at: {msmt_root}")

def generate_demo_video(base_dir: Path):
    print("Generating demo multi-camera test videos for video analysis...")
    video_dir = base_dir / "data" / "demo_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    cams = [1, 2]
    for cam_id in cams:
        video_path = video_dir / f"cam{cam_id}_demo.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 15
        w, h = 640, 360
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))

        # Simulate 2 pedestrians walking across the frame
        for frame_idx in range(90): # 6 seconds
            frame = np.full((h, w, 3), (90 + cam_id * 20, 100, 110), dtype=np.uint8)
            cv2.putText(frame, f"CAMERA {cam_id} - SIMULATED FEED", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Timestamp: 10:21:{frame_idx//15:02d}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Person 1 (PID 21) walking left to right
            p1_x = int(50 + frame_idx * 5)
            p1_y = 120
            p1_crop = generate_person_crop(21, cam_id, frame_idx, width=64, height=140)
            if 0 <= p1_x < w - 64:
                frame[p1_y:p1_y+140, p1_x:p1_x+64] = p1_crop

            # Person 2 (PID 22) walking right to left
            p2_x = int(500 - frame_idx * 4)
            p2_y = 100
            p2_crop = generate_person_crop(22, cam_id, frame_idx, width=70, height=160)
            if 0 <= p2_x < w - 70:
                frame[p2_y:p2_y+160, p2_x:p2_x+70] = p2_crop

            out.write(frame)
        out.release()
        print(f"Generated video: {video_path}")

def main():
    base_dir = Path(__file__).resolve().parent.parent
    generate_market1501_demo(base_dir)
    generate_msmt17_demo(base_dir)
    generate_demo_video(base_dir)
    print("\nAll demo benchmarks and video streams ready for validation and experiments!")

if __name__ == "__main__":
    main()
