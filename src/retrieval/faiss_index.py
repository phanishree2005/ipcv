"""
FAISS Vector Database Index for Real-time Person Re-Identification Retrieval.
Stores multi-model / fused embeddings alongside metadata:
[Global ID, Camera ID, Timestamp, Track ID, Bbox, Quality Score, Pose Vector].
Supports Top-1, Top-5, Top-10, Top-K queries with Inner Product / Cosine Similarity.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import faiss

from src.reid.base_extractor import normalize_features

class ReIDFAISSIndex:
    """
    High-performance FAISS Vector Database for Person Re-ID gallery search.
    """
    def __init__(self, embedding_dim: int = 512, index_type: str = "IndexFlatIP", use_gpu: bool = False):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu

        # Initialize FAISS Index (IndexFlatIP computes exact Inner Product / Cosine on normalized vectors)
        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)

        # In-memory metadata storage indexed by entry ID
        self.metadata: List[Dict[str, Any]] = []

    def reset(self):
        """Clears index and metadata."""
        self.index.reset()
        self.metadata.clear()

    def add_gallery_entry(self,
                          embedding: np.ndarray,
                          global_id: Union[int, str],
                          camera_id: int,
                          timestamp: float,
                          track_id: Optional[int] = None,
                          bbox: Optional[List[int]] = None,
                          quality_score: float = 1.0,
                          pose_vector: Optional[np.ndarray] = None,
                          image_path: Optional[str] = None):
        """
        Inserts a single gallery observation into the FAISS index.
        """
        emb = np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32)
        emb = normalize_features(emb)

        self.index.add(emb)
        meta_record = {
            "entry_id": len(self.metadata),
            "global_id": global_id,
            "camera_id": camera_id,
            "timestamp": timestamp,
            "track_id": track_id,
            "bbox": bbox,
            "quality_score": float(quality_score),
            "pose_vector": pose_vector.tolist() if pose_vector is not None and isinstance(pose_vector, np.ndarray) else pose_vector,
            "image_path": str(image_path) if image_path is not None else None
        }
        self.metadata.append(meta_record)

    def add_batch(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """
        Bulk insertion into index.
        """
        embs = np.ascontiguousarray(embeddings, dtype=np.float32)
        embs = normalize_features(embs)
        self.index.add(embs)
        
        start_idx = len(self.metadata)
        for i, meta in enumerate(metadata_list):
            meta_copy = dict(meta)
            meta_copy["entry_id"] = start_idx + i
            self.metadata.append(meta_copy)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves top-k closest matches for query embedding.
        Returns list of match dictionaries with similarity scores and metadata.
        """
        if self.index.ntotal == 0:
            return []

        q_emb = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        q_emb = normalize_features(q_emb)

        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx == -1:
                continue
            meta = dict(self.metadata[idx])
            results.append({
                "rank": rank,
                "similarity": float(dist), # Cosine similarity for IndexFlatIP
                "entry_id": int(idx),
                "global_id": meta.get("global_id"),
                "camera_id": meta.get("camera_id"),
                "timestamp": meta.get("timestamp"),
                "track_id": meta.get("track_id"),
                "quality_score": meta.get("quality_score"),
                "bbox": meta.get("bbox"),
                "image_path": meta.get("image_path"),
                "metadata": meta
            })
        return results

    def save(self, directory: Union[str, Path]):
        """Persists index and metadata to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        index_file = str(directory / "reid_faiss.index")
        meta_file = str(directory / "reid_metadata.json")

        faiss.write_index(self.index, index_file)
        with open(meta_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"[FAISS] Saved {self.index.ntotal} entries to {directory}")

    def load(self, directory: Union[str, Path]):
        """Loads index and metadata from disk."""
        directory = Path(directory)
        index_file = directory / "reid_faiss.index"
        meta_file = directory / "reid_metadata.json"

        if not index_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"FAISS index files not found in {directory}")

        self.index = faiss.read_index(str(index_file))
        with open(meta_file, "r") as f:
            self.metadata = json.load(f)
        print(f"[FAISS] Loaded {self.index.ntotal} entries from {directory}")

    @property
    def total_entries(self) -> int:
        return self.index.ntotal
