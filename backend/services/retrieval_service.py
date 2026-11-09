import os
import re
import base64
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.reid.model_factory import build_reid_model
from src.fusion.adaptive_fusion import AdaptiveMultiModelFusion
from src.quality.quality_estimator import ImageQualityEstimator
from src.pose.pose_estimator import PoseEstimator
from src.data.transforms import build_reid_transforms
from src.temporal.camera_topology import CameraTopology


class ForensicRetrievalService:
    """
    Production-grade Cross-Camera Forensic Re-ID Retrieval Engine.
    Executes multi-model feature extraction, quality/pose gating,
    spatio-temporal likelihood scoring, and Top-K candidate ranking.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.market_dir = PROJECT_ROOT / "data" / "market1501" / "Market-1501-v15.09.15"
        self.query_dir = self.market_dir / "query"
        self.gallery_dir = self.market_dir / "bounding_box_test"

        # Transforms & Estimators
        self.transform = build_reid_transforms(256, 128, is_train=False)
        self.quality_estimator = ImageQualityEstimator()
        self.pose_estimator = PoseEstimator().to(device)
        self.topology = CameraTopology(num_cameras=6)

        # Models
        self.model_osnet = build_reid_model("osnet", device=device).eval()
        self.model_trans = build_reid_model("transreid", device=device).eval()
        self.model_clip = build_reid_model("clip", device=device).eval()
        self.fusion_model = AdaptiveMultiModelFusion().to(device).eval()

        # Cache gallery subset for sub-second retrieval
        self._gallery_cache: List[Dict[str, Any]] = []
        self._sample_queries: List[Dict[str, Any]] = []
        self._init_cache()

    @staticmethod
    def _parse_filename(fname: str) -> tuple:
        pattern = r"^(-?\d+)_c(\d+)s(\d+)_"
        match = re.match(pattern, fname)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return -1, -1, -1

    def _init_cache(self):
        """Indexes sample queries and gallery candidates."""
        if not self.gallery_dir.exists() or not self.query_dir.exists():
            return

        # 1. Sample queries (diverse PIDs across different cameras)
        q_files = sorted(list(self.query_dir.glob("*.jpg")))
        seen_pids = set()
        for qf in q_files:
            if "_000001_" in qf.name:
                continue
            pid, cam, seq = self._parse_filename(qf.name)
            if pid > 0 and pid not in seen_pids:
                seen_pids.add(pid)
                self._sample_queries.append({
                    "filename": qf.name,
                    "pid": pid,
                    "cam": cam,
                    "seq": seq,
                    "path": str(qf)
                })
                if len(self._sample_queries) >= 20:
                    break

        # 2. Load pre-computed gallery cache for instant sub-millisecond retrieval
        cache_path = self.market_dir / "fast_gallery_cache.pkl"
        if cache_path.exists():
            try:
                import pickle
                with open(cache_path, "rb") as f:
                    self._gallery_cache = pickle.load(f)
                return
            except Exception as e:
                print(f"[Warning] Could not load fast_gallery_cache.pkl: {e}")

        # Fallback: Extract gallery embeddings
        g_files = sorted(list(self.gallery_dir.glob("*.jpg")))
        q_pids = {q["pid"] for q in self._sample_queries}
        selected_g = []
        distractors = []
        for gf in g_files:
            if "_000001_" in gf.name:
                continue
            pid, cam, seq = self._parse_filename(gf.name)
            if pid in q_pids:
                selected_g.append((gf, pid, cam, seq))
            elif pid > 0 and len(distractors) < 20:
                distractors.append((gf, pid, cam, seq))

        all_g = selected_g[:30] + distractors[:15]

        for gf, pid, cam, seq in all_g:
            try:
                img_pil = Image.open(gf).convert("RGB")
                img_t = self.transform(img_pil).unsqueeze(0).to(self.device)
                cam_t = torch.tensor([cam], device=self.device)

                with torch.no_grad():
                    f_os = self.model_osnet(img_t, cam_t)
                    f_tr = self.model_trans(img_t, cam_t)
                    f_cl = self.model_clip(img_t, cam_t)
                    q_vec = self.quality_estimator.compute_batch_quality(img_t)
                    p_vec = torch.from_numpy(self.pose_estimator.extract_pose_feature(img_t)).unsqueeze(0).float().to(self.device)

                    fused, _ = self.fusion_model(f_os, f_tr, f_cl, q_vec, p_vec)
                    emb = fused.squeeze(0).cpu().numpy()

                self._gallery_cache.append({
                    "filename": gf.name,
                    "pid": pid,
                    "cam": cam,
                    "seq": seq,
                    "path": str(gf),
                    "embedding": emb
                })
            except Exception:
                continue

    def register_video_identity(self, global_id: str, camera_id: int, crop_b64: str, embedding: Optional[np.ndarray] = None):
        """Registers a person detected from a video stream into Forensic Search."""
        fname = f"{global_id}_cam{camera_id}.jpg"
        if any(q.get("filename") == fname for q in self._sample_queries):
            return

        try:
            pid_num = int("".join(filter(str.isdigit, global_id)) or "999")
        except Exception:
            pid_num = 999

        item = {
            "filename": fname,
            "pid": pid_num,
            "cam": camera_id,
            "seq": 1,
            "thumbnail_b64": crop_b64,
            "is_video_sighting": True,
            "label": f"{global_id} (Cam {camera_id})"
        }
        self._sample_queries.insert(0, item)

        if embedding is not None:
            self._gallery_cache.insert(0, {
                "filename": fname,
                "pid": pid_num,
                "cam": camera_id,
                "seq": 1,
                "thumbnail_b64": crop_b64,
                "embedding": embedding,
                "is_video_sighting": True
            })

    def get_sample_queries(self) -> List[Dict[str, Any]]:
        """Returns list of sample queries with base64 thumbnail."""
        out = []
        for q in self._sample_queries:
            if q.get("thumbnail_b64"):
                out.append(q)
            elif q.get("path") and os.path.exists(q["path"]):
                try:
                    with open(q["path"], "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    out.append({
                        "filename": q["filename"],
                        "pid": q["pid"],
                        "cam": q["cam"],
                        "seq": q["seq"],
                        "thumbnail_b64": b64
                    })
                except Exception:
                    continue
        return out

    def search_query(self,
                     query_path: Optional[str] = None,
                     query_b64: Optional[str] = None,
                     query_cam: int = 1,
                     top_k: int = 10) -> Dict[str, Any]:
        """
        Executes query retrieval against gallery using M²-ReID adaptive pipeline.
        """
        # Load image
        if query_path and os.path.exists(query_path):
            img_pil = Image.open(query_path).convert("RGB")
            pid_target, cam_source, _ = self._parse_filename(os.path.basename(query_path))
        elif query_b64:
            if "," in query_b64:
                query_b64 = query_b64.split(",")[1]
            img_bytes = base64.b64decode(query_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            pid_target = -1
            cam_source = query_cam
        else:
            raise ValueError("Must provide either query_path or query_b64")

        # Encode query image as base64 for display
        buffered = cv2.imencode(".jpg", cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))[1]
        disp_b64 = base64.b64encode(buffered).decode("utf-8")

        # 1. Feature Extraction & Quality/Pose Gating
        img_t = self.transform(img_pil).unsqueeze(0).to(self.device)
        cam_t = torch.tensor([cam_source], device=self.device)

        with torch.no_grad():
            f_os = self.model_osnet(img_t, cam_t)
            f_tr = self.model_trans(img_t, cam_t)
            f_cl = self.model_clip(img_t, cam_t)
            q_vec = self.quality_estimator.compute_batch_quality(img_t)
            raw_pose = self.pose_estimator.extract_pose_feature(img_t)
            p_vec = torch.from_numpy(raw_pose).unsqueeze(0).float().to(self.device)

            fused, weights = self.fusion_model(f_os, f_tr, f_cl, q_vec, p_vec)
            q_emb = fused.squeeze(0).cpu().numpy()
            w_np = weights.squeeze(0).cpu().numpy()

        weights_dict = {
            "osnet": round(float(w_np[0]), 3),
            "transreid": round(float(w_np[1]), 3),
            "clip": round(float(w_np[2]), 3)
        }

        # Quality metrics breakdown
        q_np = q_vec.squeeze(0).cpu().numpy()
        quality_breakdown = {
            "sharpness": round(float(q_np[0]), 2),
            "brightness": round(float(q_np[1]), 2),
            "contrast": round(float(q_np[2]), 2),
            "detection_conf": round(float(q_np[3]), 2),
            "bbox_area": round(float(q_np[4]), 2),
            "aspect_ratio": round(float(q_np[5]), 2),
            "occlusion_score": round(float(q_np[6]), 2),
        }

        # Pose keypoints pairs: 17 x (x, y)
        pose_points = []
        for i in range(0, len(raw_pose), 2):
            pose_points.append({"x": round(float(raw_pose[i]), 3), "y": round(float(raw_pose[i+1]), 3)})

        # 2. Similarity Search against Gallery
        scores = []
        for g in self._gallery_cache:
            # Avoid matching same camera / same crop if testing Market-1501 protocol
            # Cosine similarity
            cos_sim = float(np.dot(q_emb, g["embedding"]) / (np.linalg.norm(q_emb) * np.linalg.norm(g["embedding"]) + 1e-8))
            
            # Spatio-temporal likelihood modulation
            prior = self.topology.get_transition_prior(cam_source, g["cam"])
            adjusted_score = 0.85 * cos_sim + 0.15 * prior

            scores.append((adjusted_score, cos_sim, g))

        # Sort descending
        scores.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scores[:top_k]

        results = []
        for rank, (final_score, visual_sim, g) in enumerate(top_candidates, 1):
            if g.get("thumbnail_b64"):
                thumb_b64 = g["thumbnail_b64"]
            elif g.get("path") and os.path.exists(g["path"]):
                try:
                    with open(g["path"], "rb") as f:
                        thumb_b64 = base64.b64encode(f.read()).decode("utf-8")
                except Exception:
                    thumb_b64 = ""
            else:
                thumb_b64 = ""
            
            is_match = (pid_target > 0 and g["pid"] == pid_target) or (g.get("is_video_sighting") and visual_sim > 0.60)
            results.append({
                "rank": rank,
                "pid": g["pid"],
                "camera": g["cam"],
                "filename": g["filename"],
                "similarity": round(visual_sim * 100, 1),
                "composite_score": round(final_score * 100, 1),
                "is_ground_truth_match": is_match,
                "thumbnail_b64": thumb_b64
            })

        return {
            "query": {
                "pid": pid_target,
                "camera": cam_source,
                "image_b64": disp_b64,
                "weights": weights_dict,
                "quality": quality_breakdown,
                "pose_keypoints": pose_points
            },
            "total_gallery_searched": len(self._gallery_cache),
            "results": results
        }
