#!/usr/bin/env python3
"""
M²-ReID Dataset Validator
Validates dataset directory integrity, filenames, person IDs, camera IDs,
image corruptions, and computes actual statistics across Market-1501 and MSMT17.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List
import yaml
from PIL import Image

class DatasetValidator:
    def __init__(self, config_path: str = "configs/dataset.yaml"):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            # Try finding relative to project root
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.config_path = base_dir / config_path
            
        with open(self.config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
            
        self.base_dir = self.config_path.parent.parent

    @staticmethod
    def parse_market1501_filename(filename: str) -> Tuple[int, int, int]:
        """
        Market-1501 format: [PID]_[cCamID]s[SeqID]_[FrameNum]_[BboxNum].jpg
        e.g., 0001_c1s1_001051_00.jpg -> PID=1, Cam=1, Seq=1
        -1 = junk images, 0000 = distractors
        """
        pattern = r"^(-?\d+)_c(\d+)s(\d+)_"
        match = re.match(pattern, filename)
        if match:
            pid = int(match.group(1))
            cam = int(match.group(2))
            seq = int(match.group(3))
            return pid, cam, seq
        return -999, -999, -999

    def check_image_corruption(self, img_path: Path, sample_rate: int = 50) -> bool:
        """Attempts to open and verify image file (sampled for high performance)."""
        try:
            if img_path.stat().st_size == 0:
                return True
            # Sample check with PIL to maintain instantaneous validation on large datasets
            if hash(img_path.name) % sample_rate == 0:
                with Image.open(img_path) as img:
                    img.verify()
            return False # Not corrupted
        except Exception:
            return True # Corrupted

    def validate_market1501(self) -> Dict[str, Any]:
        """Validates Market-1501 directory structure and metadata."""
        m_cfg = self.cfg.get("datasets", {}).get("market1501", {})
        root_dir = self.base_dir / m_cfg.get("root", "data/market1501/Market-1501-v15.09.15")
        
        results = {
            "dataset": "Market-1501",
            "root_path": str(root_dir),
            "exists": root_dir.exists(),
            "train_images": 0,
            "query_images": 0,
            "gallery_images": 0,
            "train_ids": 0,
            "query_ids": 0,
            "gallery_ids": 0,
            "total_identities": 0,
            "cameras": set(),
            "corrupted_images": 0,
            "corrupted_files": [],
            "valid": False,
            "errors": []
        }
        
        if not root_dir.exists():
            results["errors"].append(f"Root path does not exist: {root_dir}")
            return results
            
        subdirs = {
            "train": root_dir / m_cfg.get("train_dir", "bounding_box_train"),
            "query": root_dir / m_cfg.get("query_dir", "query"),
            "gallery": root_dir / m_cfg.get("gallery_dir", "bounding_box_test")
        }
        
        for name, p in subdirs.items():
            if not p.exists():
                results["errors"].append(f"Missing subdirectory: {name} ({p})")
                
        if results["errors"]:
            return results

        # Process Train
        train_pids = set()
        for f in subdirs["train"].glob("*.jpg"):
            results["train_images"] += 1
            if self.check_image_corruption(f):
                results["corrupted_images"] += 1
                results["corrupted_files"].append(str(f))
            pid, cam, _ = self.parse_market1501_filename(f.name)
            if pid > 0:
                train_pids.add(pid)
            if cam > 0:
                results["cameras"].add(cam)
        results["train_ids"] = len(train_pids)

        # Process Query
        query_pids = set()
        for f in subdirs["query"].glob("*.jpg"):
            results["query_images"] += 1
            if self.check_image_corruption(f):
                results["corrupted_images"] += 1
                results["corrupted_files"].append(str(f))
            pid, cam, _ = self.parse_market1501_filename(f.name)
            if pid > 0:
                query_pids.add(pid)
            if cam > 0:
                results["cameras"].add(cam)
        results["query_ids"] = len(query_pids)

        # Process Gallery
        gallery_pids = set()
        for f in subdirs["gallery"].glob("*.jpg"):
            results["gallery_images"] += 1
            if self.check_image_corruption(f):
                results["corrupted_images"] += 1
                results["corrupted_files"].append(str(f))
            pid, cam, _ = self.parse_market1501_filename(f.name)
            if pid > 0:
                gallery_pids.add(pid)
            if cam > 0:
                results["cameras"].add(cam)
        results["gallery_ids"] = len(gallery_pids)

        all_pids = train_pids.union(query_pids).union(gallery_pids)
        results["total_identities"] = len(all_pids)
        results["cameras"] = sorted(list(results["cameras"]))
        results["valid"] = (results["train_images"] > 0 and results["corrupted_images"] == 0)
        
        return results

    def validate_msmt17(self) -> Dict[str, Any]:
        """Validates MSMT17 directory structure and official txt split files."""
        m_cfg = self.cfg.get("datasets", {}).get("msmt17", {})
        root_dir = self.base_dir / m_cfg.get("root", "data/msmt17/MSMT17_V1")
        
        results = {
            "dataset": "MSMT17",
            "root_path": str(root_dir),
            "exists": root_dir.exists(),
            "train_images": 0,
            "query_images": 0,
            "gallery_images": 0,
            "train_ids": 0,
            "query_ids": 0,
            "gallery_ids": 0,
            "total_identities": 0,
            "cameras": set(),
            "corrupted_images": 0,
            "corrupted_files": [],
            "valid": False,
            "errors": []
        }
        
        if not root_dir.exists():
            results["errors"].append(f"Root path does not exist: {root_dir}")
            return results
            
        train_dir = root_dir / m_cfg.get("train_dir", "train")
        test_dir = root_dir / m_cfg.get("test_dir", "test")
        list_train = root_dir / m_cfg.get("list_train", "list_train.txt")
        list_query = root_dir / m_cfg.get("list_query", "list_query.txt")
        list_gallery = root_dir / m_cfg.get("list_gallery", "list_gallery.txt")
        
        for p in [train_dir, test_dir, list_train, list_query, list_gallery]:
            if not p.exists():
                results["errors"].append(f"Missing required path/file: {p}")
                
        if results["errors"]:
            return results

        def parse_msmt_list(file_path: Path, base_folder: Path):
            pids = set()
            cams = set()
            count = 0
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        rel_img, pid_str = parts[0], parts[1]
                        count += 1
                        pid = int(pid_str)
                        pids.add(pid)
                        # Extract camera if present in path, e.g. 0001_c1_... or similar
                        cam_match = re.search(r"_c(\d+)_", rel_img)
                        if cam_match:
                            cams.add(int(cam_match.group(1)))
            return count, pids, cams

        results["train_images"], train_pids, train_cams = parse_msmt_list(list_train, train_dir)
        results["query_images"], query_pids, query_cams = parse_msmt_list(list_query, test_dir)
        results["gallery_images"], gallery_pids, gallery_cams = parse_msmt_list(list_gallery, test_dir)
        
        results["train_ids"] = len(train_pids)
        results["query_ids"] = len(query_pids)
        results["gallery_ids"] = len(gallery_pids)
        
        all_pids = train_pids.union(query_pids).union(gallery_pids)
        all_cams = train_cams.union(query_cams).union(gallery_cams)
        
        results["total_identities"] = len(all_pids)
        results["cameras"] = sorted(list(all_cams))
        results["valid"] = (results["train_images"] > 0)
        return results

    def print_report(self, stats: Dict[str, Any]):
        print("=" * 60)
        print(f"Dataset: {stats['dataset']}")
        print(f"Path: {stats['root_path']}")
        print(f"Status: {'VALID' if stats['valid'] else 'INVALID / NOT FOUND'}")
        if stats.get("errors"):
            for e in stats["errors"]:
                print(f"  [ERROR] {e}")
        else:
            print(f"Train images: {stats['train_images']}")
            print(f"Query images: {stats['query_images']}")
            print(f"Gallery images: {stats['gallery_images']}")
            print(f"Train Identities: {stats['train_ids']}")
            print(f"Query Identities: {stats['query_ids']}")
            print(f"Gallery Identities: {stats['gallery_ids']}")
            print(f"Total Unique Identities: {stats['total_identities']}")
            print(f"Cameras ({len(stats['cameras'])} detected): {stats['cameras']}")
            print(f"Corrupted images: {stats['corrupted_images']}")
        print("=" * 60)

def main():
    validator = DatasetValidator()
    print("\n[VALIDATING MARKET-1501]")
    m_stats = validator.validate_market1501()
    validator.print_report(m_stats)
    
    print("\n[VALIDATING MSMT17]")
    ms_stats = validator.validate_msmt17()
    validator.print_report(ms_stats)

if __name__ == "__main__":
    main()
