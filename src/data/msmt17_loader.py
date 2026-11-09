"""
Dataset and DataLoader classes for MSMT17 (Multi-Scene Multi-Time) benchmark.
Parses official list_train.txt, list_val.txt, list_query.txt, list_gallery.txt.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.transforms import build_reid_transforms
from src.data.market1501_loader import ReIDImageDataset, RandomIdentitySampler

class MSMT17DataLoader:
    def __init__(self, root_dir: str, height: int = 256, width: int = 128,
                 batch_size: int = 32, num_instances: int = 4, num_workers: int = 2):
        self.root_dir = Path(root_dir)
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_workers = num_workers

        self.train_dir = self.root_dir / "train"
        self.test_dir = self.root_dir / "test"
        self.list_train = self.root_dir / "list_train.txt"
        self.list_val = self.root_dir / "list_val.txt"
        self.list_query = self.root_dir / "list_query.txt"
        self.list_gallery = self.root_dir / "list_gallery.txt"

        self.train_transform = build_reid_transforms(height, width, is_train=True)
        self.test_transform = build_reid_transforms(height, width, is_train=False)

    def _parse_list(self, list_file: Path, base_dir: Path, is_train: bool = False) -> Tuple[List[Tuple[str, int, int, int]], int, int]:
        if not list_file.exists():
            raise FileNotFoundError(f"MSMT17 list file not found: {list_file}")

        dataset = []
        pids = set()
        cams = set()

        with open(list_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    rel_path, pid_str = parts[0], parts[1]
                    img_path = base_dir / rel_path
                    pid = int(pid_str)
                    
                    # Extract camera ID from filename, e.g. 0001_c1_... or 0001_c01_...
                    cam_id = 1
                    parts_fn = rel_path.split("_")
                    for p in parts_fn:
                        if p.startswith("c") and p[1:].isdigit():
                            cam_id = int(p[1:])
                            break
                    
                    pids.add(pid)
                    cams.add(cam_id)
                    dataset.append((str(img_path), pid, cam_id, 0))

        if is_train:
            pid2label = {pid: idx for idx, pid in enumerate(sorted(list(pids)))}
            relabelled = [(p, pid2label[pid], c, s) for (p, pid, c, s) in dataset]
            return relabelled, len(pids), len(cams)

        return dataset, len(pids), len(cams)

    def get_dataloaders(self) -> Dict[str, Any]:
        if not self.root_dir.exists():
            raise RuntimeError(f"MSMT17 root directory not found at: {self.root_dir}")

        train_data, num_train_pids, num_train_cams = self._parse_list(self.list_train, self.train_dir, is_train=True)
        query_data, num_query_pids, num_query_cams = self._parse_list(self.list_query, self.test_dir, is_train=False)
        gallery_data, num_gallery_pids, num_gallery_cams = self._parse_list(self.list_gallery, self.test_dir, is_train=False)

        train_dataset = ReIDImageDataset(train_data, transform=self.train_transform)
        query_dataset = ReIDImageDataset(query_data, transform=self.test_transform)
        gallery_dataset = ReIDImageDataset(gallery_data, transform=self.test_transform)

        train_sampler = RandomIdentitySampler(train_data, batch_size=self.batch_size, num_instances=self.num_instances)

        pin_mem = torch.cuda.is_available()
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=pin_mem,
            drop_last=True
        )

        query_loader = DataLoader(
            query_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_mem
        )

        gallery_loader = DataLoader(
            gallery_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_mem
        )

        return {
            "train_loader": train_loader,
            "query_loader": query_loader,
            "gallery_loader": gallery_loader,
            "num_train_pids": num_train_pids,
            "num_train_cams": num_train_cams,
            "query_data": query_data,
            "gallery_data": gallery_data,
            "train_data": train_data
        }
