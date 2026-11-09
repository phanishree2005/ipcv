"""
Re-ID Benchmark Evaluator Engine.
Extracts representations across Query and Gallery sets, executes benchmark evaluation protocols,
and records Rank-1, Rank-5, Rank-10, mAP, and efficiency metrics.
"""

import time
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

from src.reid.base_extractor import normalize_features
from src.evaluation.metrics import evaluate_rank_and_map, SystemResourceProfiler


class ReIDEvaluator:
    """
    Standard query-gallery benchmark evaluator.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device

    @torch.no_grad()
    def extract_features(self,
                         model,
                         dataloader,
                         is_fusion: bool = False,
                         extractor_osnet = None,
                         extractor_trans = None,
                         extractor_clip = None,
                         quality_estimator = None,
                         pose_estimator = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Extracts features, PIDs, and Camera IDs from a DataLoader.
        """
        model.eval()
        feats, pids, camids, img_paths = [], [], [], []

        for imgs, batch_pids, batch_cams, _, paths in dataloader:
            imgs = imgs.to(self.device)
            batch_cams_tensor = batch_cams.to(self.device)

            if is_fusion:
                f_os = extractor_osnet(imgs, batch_cams_tensor)
                f_tr = extractor_trans(imgs, batch_cams_tensor)
                f_cl = extractor_clip(imgs, batch_cams_tensor)
                q_vec = quality_estimator.compute_batch_quality(imgs)
                p_vec = torch.from_numpy(
                    np.stack([pose_estimator.extract_pose_feature(img) for img in imgs])
                ).float().to(self.device)

                fused_feat, _ = model(f_os, f_tr, f_cl, q_vec, p_vec)
                feat = fused_feat
            else:
                feat = model(imgs, batch_cams_tensor)

            feats.append(feat.detach().cpu().numpy())
            pids.extend(batch_pids.numpy())
            camids.extend(batch_cams.numpy())
            img_paths.extend(paths)

        feats = np.concatenate(feats, axis=0)
        pids = np.array(pids)
        camids = np.array(camids)
        return normalize_features(feats), pids, camids, img_paths

    def evaluate(self,
                 query_feats: np.ndarray,
                 gallery_feats: np.ndarray,
                 q_pids: np.ndarray,
                 g_pids: np.ndarray,
                 q_camids: np.ndarray,
                 g_camids: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates extracted features against official Re-ID protocol.
        """
        # Cosine distance: 1.0 - inner product
        distmat = 1.0 - np.dot(query_feats, gallery_feats.T)
        distmat = np.clip(distmat, 0.0, 2.0)

        cmc, mAP = evaluate_rank_and_map(distmat, q_pids, g_pids, q_camids, g_camids)

        r1 = float(cmc[0] * 100.0) if len(cmc) > 0 else 0.0
        r5 = float(cmc[4] * 100.0) if len(cmc) > 4 else (float(cmc[-1] * 100.0) if len(cmc) > 0 else 0.0)
        r10 = float(cmc[9] * 100.0) if len(cmc) > 9 else (float(cmc[-1] * 100.0) if len(cmc) > 0 else 0.0)
        map_score = float(mAP * 100.0)

        return {
            "rank1": float(round(r1, 2)),
            "rank5": float(round(r5, 2)),
            "rank10": float(round(r10, 2)),
            "mAP": float(round(map_score, 2)),
            "cmc_curve": cmc.tolist(),
            "distmat": distmat
        }
