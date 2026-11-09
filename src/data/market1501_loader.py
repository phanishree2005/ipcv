"""
Dataset and DataLoader classes for Market-1501 benchmark.
Implements RandomIdentitySampler (P x K batching) and standard validation/query/gallery loaders.
"""

import os
import re
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy

from src.data.transforms import build_reid_transforms


class ReIDImageDataset(Dataset):
    """
    Generic Re-ID dataset supporting (image_path, person_id, camera_id, seq_id).
    """
    def __init__(self, dataset: List[Tuple[str, int, int, int]], transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, seqid = self.dataset[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, seqid, img_path


class RandomIdentitySampler(Sampler):
    """
    Randomly samples N identities, then K instances for each identity in each batch.
    Batch size = num_instances * num_pids_per_batch
    Essential for Triplet Loss mining.
    """
    def __init__(self, data_source: List[Tuple[str, int, int, int]], batch_size: int, num_instances: int):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class Market1501DataLoader:
    """
    Parser and DataLoader builder for Market-1501.
    """
    def __init__(self, root_dir: str, height: int = 256, width: int = 128,
                 batch_size: int = 32, num_instances: int = 4, num_workers: int = 2):
        self.root_dir = Path(root_dir)
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_workers = num_workers

        self.train_dir = self.root_dir / "bounding_box_train"
        self.query_dir = self.root_dir / "query"
        self.gallery_dir = self.root_dir / "bounding_box_test"

        self.train_transform = build_reid_transforms(height, width, is_train=True)
        self.test_transform = build_reid_transforms(height, width, is_train=False)

        self._check_before_run()

    def _check_before_run(self):
        if not self.root_dir.exists():
            raise RuntimeError(f"'{self.root_dir}' is not found. Please verify data path or run check_datasets.py.")

    @staticmethod
    def _process_dir(dir_path: Path, is_train: bool = True) -> Tuple[List[Tuple[str, int, int, int]], int, int]:
        pattern = re.compile(r"^(-?\d+)_c(\d+)s(\d+)_")
        img_paths = list(dir_path.glob("*.jpg"))
        dataset = []
        pid_container = set()
        cam_container = set()

        for img_path in img_paths:
            match = pattern.match(img_path.name)
            if not match:
                continue
            pid, camid, seqid = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if pid == -1 or pid == 0:
                if is_train:
                    continue  # ignore junk in train
            pid_container.add(pid)
            cam_container.add(camid)
            dataset.append((str(img_path), pid, camid, seqid))

        # Relabel PIDs for training if needed
        if is_train:
            pid2label = {pid: label for label, pid in enumerate(sorted(list(pid_container)))}
            relabelled_dataset = []
            for img_path, pid, camid, seqid in dataset:
                relabelled_dataset.append((img_path, pid2label[pid], camid, seqid))
            return relabelled_dataset, len(pid_container), len(cam_container)

        return dataset, len(pid_container), len(cam_container)

    def get_dataloaders(self) -> Dict[str, Any]:
        train_data, num_train_pids, num_train_cams = self._process_dir(self.train_dir, is_train=True)
        query_data, num_query_pids, num_query_cams = self._process_dir(self.query_dir, is_train=False)
        gallery_data, num_gallery_pids, num_gallery_cams = self._process_dir(self.gallery_dir, is_train=False)

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
