"""
Automated Error Analysis and Failure Case Categorization Engine.
Analyzes false matches (Rank-1 errors) and categorizes failure modes into:
1. Similar appearance / visual ambiguity
2. Heavy occlusion
3. Low resolution / distant capture
4. Lighting / illumination shift
5. Extreme pose / viewpoint difference
6. Cross-camera domain shift
Generates structured JSON error reports and visual side-by-side failure cards.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from PIL import Image

from src.quality.quality_estimator import ImageQualityEstimator
from src.pose.pose_estimator import PoseEstimator


class ErrorAnalyzer:
    """
    Diagnoses and categorizes Re-ID ranking failures.
    """
    def __init__(self, output_dir: str = "results/error_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quality_estimator = ImageQualityEstimator()

    def diagnose_failure_cause(self,
                               q_img_path: str,
                               top1_img_path: str,
                               q_cam: int,
                               top1_cam: int) -> Dict[str, Any]:
        """
        Extracts diagnostic signals from query and false top-1 gallery image
        to automatically classify the primary root cause of the error.
        """
        try:
            q_img = cv2.imread(q_img_path)
            t_img = cv2.imread(top1_img_path)
        except Exception:
            return {"primary_cause": "Cross-camera ambiguity", "confidence": 0.60, "notes": "Image read error"}

        if q_img is None or t_img is None:
            return {"primary_cause": "Cross-camera ambiguity", "confidence": 0.60, "notes": "Missing image"}

        q_qual = self.quality_estimator.extract_quality_vector(cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB))
        t_qual = self.quality_estimator.extract_quality_vector(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB))

        q_blur, q_bright, q_contrast, _, _, _, q_occlusion = q_qual
        t_blur, t_bright, t_contrast, _, _, _, t_occlusion = t_qual

        q_pose = PoseEstimator().extract_pose_feature(q_img)
        t_pose = PoseEstimator().extract_pose_feature(t_img)
        pose_sim = PoseEstimator.compute_pose_similarity(q_pose, t_pose)

        # Categorization Decision Rules
        causes = []
        if q_occlusion > 0.60 or t_occlusion > 0.60:
            causes.append(("Occlusion", 0.85, f"High occlusion score detected (Q: {q_occlusion:.2f}, Top1: {t_occlusion:.2f})"))
        
        if q_blur < 0.25 or t_blur < 0.25:
            causes.append(("Low resolution", 0.80, f"Low sharpness/blur metric (Q: {q_blur:.2f}, Top1: {t_blur:.2f})"))

        if abs(q_bright - t_bright) > 0.35 or q_bright < 0.25 or t_bright < 0.25:
            causes.append(("Lighting", 0.75, f"Drastic lighting disparity between cameras (Q: {q_bright:.2f}, Top1: {t_bright:.2f})"))

        if pose_sim < 0.40:
            causes.append(("Pose difference", 0.70, f"Major viewpoint / pose orientation shift (Pose Sim: {pose_sim:.2f})"))

        if q_cam != top1_cam:
            causes.append(("Cross-camera ambiguity", 0.65, f"Cross camera shift from c{q_cam} to c{top1_cam}"))

        # Default fallback
        causes.append(("Similar appearance", 0.60, "Visually confusing clothing color distribution"))

        # Sort by diagnostic confidence
        causes.sort(key=lambda x: x[1], reverse=True)
        primary = causes[0]

        return {
            "primary_cause": primary[0],
            "confidence": primary[1],
            "notes": primary[2],
            "diagnostics": {
                "q_blur": float(q_blur),
                "q_bright": float(q_bright),
                "q_occlusion": float(q_occlusion),
                "pose_similarity": float(pose_sim),
                "cam_transition": f"c{q_cam} -> c{top1_cam}"
            }
        }

    def analyze_dataset_errors(self,
                               distmat: np.ndarray,
                               q_pids: np.ndarray,
                               g_pids: np.ndarray,
                               q_camids: np.ndarray,
                               g_camids: np.ndarray,
                               q_paths: List[str],
                               g_paths: List[str],
                               max_saved_errors: int = 30) -> Dict[str, Any]:
        """
        Scans all queries, finds Rank-1 misidentifications, diagnoses causes,
        and saves visual error report cards.
        """
        indices = np.argsort(distmat, axis=1)
        error_cases = []
        cause_distribution = {}

        for q_idx in range(len(q_pids)):
            q_pid = q_pids[q_idx]
            q_cam = q_camids[q_idx]
            order = indices[q_idx]

            # Filter out junk matches (same PID and same Cam)
            valid_mask = ~((g_pids[order] == q_pid) & (g_camids[order] == q_cam))
            valid_order = order[valid_mask]

            if len(valid_order) == 0:
                continue

            top1_idx = valid_order[0]
            top1_pid = g_pids[top1_idx]
            top1_cam = g_camids[top1_idx]

            # Rank-1 Failure (top-1 is different person identity)
            if top1_pid != q_pid:
                diag = self.diagnose_failure_cause(q_paths[q_idx], g_paths[top1_idx], q_cam, top1_cam)
                cause = diag["primary_cause"]
                cause_distribution[cause] = cause_distribution.get(cause, 0) + 1

                if len(error_cases) < max_saved_errors:
                    error_cases.append({
                        "query_index": int(q_idx),
                        "query_pid": int(q_pid),
                        "query_cam": int(q_cam),
                        "query_path": q_paths[q_idx],
                        "false_top1_pid": int(top1_pid),
                        "false_top1_cam": int(top1_cam),
                        "false_top1_path": g_paths[top1_idx],
                        "distance": float(distmat[q_idx, top1_idx]),
                        "similarity": float(1.0 - distmat[q_idx, top1_idx]),
                        "primary_cause": cause,
                        "diagnosis": diag
                    })

        report = {
            "total_queries": int(len(q_pids)),
            "total_rank1_errors": int(sum(cause_distribution.values())),
            "error_rate": float(sum(cause_distribution.values()) / max(1, len(q_pids))),
            "cause_distribution": cause_distribution,
            "sample_errors": error_cases
        }

        report_file = self.output_dir / "error_analysis_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"[ERROR ANALYSIS] Identified {len(error_cases)} error cases. Report saved to {report_file}")
        return report
