"""
Global Cross-Camera Identity Manager for Person Re-Identification.
Maintains global identity registry, camera transit timelines, representative high-quality gallery exemplars,
and associates incoming camera tracklets with global person IDs.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime

from src.reid.base_extractor import normalize_features
from src.retrieval.faiss_index import ReIDFAISSIndex
from src.retrieval.reranking import MultiContextReRanker
from src.identity.calibration import ConfidenceCalibrator


class IdentityRecord:
    """
    Represents a unique global person identity (e.g. 'P017').
    """
    def __init__(self, global_id: str, initial_embedding: np.ndarray, initial_camera: int,
                 initial_time: float, track_id: int, quality_score: float = 0.8,
                 crop: Optional[np.ndarray] = None, max_exemplars: int = 8):
        self.global_id = global_id
        self.exemplar_embeddings: List[np.ndarray] = [normalize_features(initial_embedding)]
        self.exemplar_qualities: List[float] = [quality_score]
        self.exemplar_crops: List[Optional[np.ndarray]] = [crop]
        self.max_exemplars = max_exemplars

        # Spatial-temporal trajectory sightings: list of {camera_id, timestamp, track_id, bbox, confidence}
        self.timeline: List[Dict[str, Any]] = [{
            "camera_id": initial_camera,
            "timestamp": initial_time,
            "track_id": track_id,
            "quality_score": quality_score,
            "datetime_str": datetime.fromtimestamp(initial_time).strftime("%H:%M:%S") if initial_time > 100000 else f"{int(initial_time//60):02d}:{int(initial_time%60):02d}"
        }]

    @property
    def representative_embedding(self) -> np.ndarray:
        """Quality-weighted average embedding of all exemplars."""
        q_weights = np.array(self.exemplar_qualities, dtype=np.float32)
        q_weights = q_weights / (np.sum(q_weights) + 1e-8)
        stacked = np.stack(self.exemplar_embeddings, axis=0)
        rep = np.sum(stacked * q_weights[:, None], axis=0)
        return normalize_features(rep)

    def add_observation(self,
                        embedding: np.ndarray,
                        camera_id: int,
                        timestamp: float,
                        track_id: int,
                        quality_score: float,
                        crop: Optional[np.ndarray] = None,
                        bbox: Optional[List[int]] = None,
                        min_quality_for_gallery: float = 0.50):
        """
        Records sighting and conditionally updates exemplar gallery if quality is high.
        """
        # Record timeline event
        time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S") if timestamp > 100000 else f"{int(timestamp//60):02d}:{int(timestamp%60):02d}"
        self.timeline.append({
            "camera_id": camera_id,
            "timestamp": timestamp,
            "track_id": track_id,
            "bbox": bbox,
            "quality_score": quality_score,
            "datetime_str": time_str
        })

        # Quality-filtered exemplar addition (Do not blindly add noisy crops)
        if quality_score >= min_quality_for_gallery:
            norm_emb = normalize_features(embedding)
            if len(self.exemplar_embeddings) < self.max_exemplars:
                self.exemplar_embeddings.append(norm_emb)
                self.exemplar_qualities.append(quality_score)
                self.exemplar_crops.append(crop)
            else:
                # Replace lowest quality exemplar if new one is better
                min_idx = int(np.argmin(self.exemplar_qualities))
                if quality_score > self.exemplar_qualities[min_idx]:
                    self.exemplar_embeddings[min_idx] = norm_emb
                    self.exemplar_qualities[min_idx] = quality_score
                    self.exemplar_crops[min_idx] = crop


class GlobalIdentityManager:
    """
    Manages global person identities across the entire multi-camera network.
    """
    def __init__(self,
                 faiss_index: Optional[ReIDFAISSIndex] = None,
                 reranker: Optional[MultiContextReRanker] = None,
                 calibrator: Optional[ConfidenceCalibrator] = None,
                 match_threshold: float = 0.65,
                 max_exemplars: int = 8):
        self.faiss_index = faiss_index or ReIDFAISSIndex()
        self.reranker = reranker or MultiContextReRanker()
        self.calibrator = calibrator or ConfidenceCalibrator()
        self.match_threshold = match_threshold
        self.max_exemplars = max_exemplars

        self.identities: Dict[str, IdentityRecord] = {}
        self.id_counter = 0

    def get_or_create_identity(self,
                               embedding: np.ndarray,
                               camera_id: int,
                               timestamp: float,
                               track_id: int,
                               quality_score: float = 0.8,
                               pose_vector: Optional[np.ndarray] = None,
                               crop: Optional[np.ndarray] = None,
                               bbox: Optional[List[int]] = None) -> Tuple[str, float, float, Dict[str, Any]]:
        """
        Assigns or registers a global identity for an observation.
        Returns (global_id, similarity_score, calibrated_confidence, match_details).
        """
        # Query FAISS index for candidate matches
        candidates = self.faiss_index.search(embedding, top_k=5)

        query_info = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "quality_score": quality_score,
            "pose_vector": pose_vector
        }

        best_gid = None
        best_sim = 0.0
        calibrated_conf = 0.0
        match_details = {}

        if len(candidates) > 0:
            reranked = self.reranker.rerank_candidates(query_info, candidates)
            top_cand = reranked[0]
            best_sim = top_cand.get("final_score", top_cand["similarity"])
            calibrated_conf = self.calibrator.calibrate(best_sim)

            if best_sim >= self.match_threshold:
                best_gid = top_cand["global_id"]
                match_details = top_cand

        if best_gid is not None and best_gid in self.identities:
            # Match confirmed: update existing identity
            identity = self.identities[best_gid]
            identity.add_observation(
                embedding=embedding,
                camera_id=camera_id,
                timestamp=timestamp,
                track_id=track_id,
                quality_score=quality_score,
                crop=crop,
                bbox=bbox
            )
        else:
            # Create new Global Identity (e.g. P001, P002, ...)
            self.id_counter += 1
            best_gid = f"P{self.id_counter:03d}"
            identity = IdentityRecord(
                global_id=best_gid,
                initial_embedding=embedding,
                initial_camera=camera_id,
                initial_time=timestamp,
                track_id=track_id,
                quality_score=quality_score,
                crop=crop,
                max_exemplars=self.max_exemplars
            )
            self.identities[best_gid] = identity
            best_sim = 1.0
            calibrated_conf = self.calibrator.calibrate(best_sim)

        # Update FAISS gallery with representative embedding
        if quality_score >= 0.50:
            self.faiss_index.add_gallery_entry(
                embedding=embedding,
                global_id=best_gid,
                camera_id=camera_id,
                timestamp=timestamp,
                track_id=track_id,
                bbox=bbox,
                quality_score=quality_score,
                pose_vector=pose_vector
            )

        return best_gid, float(best_sim), float(calibrated_conf), match_details

    def get_identity_summary(self) -> List[Dict[str, Any]]:
        """Returns summary list of all registered global identities and sighting trajectories."""
        summary = []
        for gid, rec in self.identities.items():
            cams_visited = [s["camera_id"] for s in rec.timeline]
            timeline_str = " -> ".join([f"Cam {s['camera_id']} ({s['datetime_str']})" for s in rec.timeline])
            qualities = [q for q in rec.exemplar_qualities if q is not None]
            base_q = float(np.mean(qualities)) if qualities else 0.75
            # Dynamic confidence per identity based on feature quality and sighting stability
            conf = float(np.clip(base_q * (0.82 + 0.16 * min(1.0, len(rec.timeline) / 5.0)), 0.65, 0.97))
            summary.append({
                "global_id": gid,
                "num_sightings": len(rec.timeline),
                "num_exemplars": len(rec.exemplar_embeddings),
                "cameras_visited": sorted(list(set(cams_visited))),
                "timeline_str": timeline_str,
                "first_seen": rec.timeline[0]["datetime_str"],
                "last_seen": rec.timeline[-1]["datetime_str"],
                "best_crop": rec.exemplar_crops[0] if len(rec.exemplar_crops) > 0 else None,
                "confidence": round(conf, 3)
            })
        return summary
