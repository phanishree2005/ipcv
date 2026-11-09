"""
Comprehensive Evaluation Metrics Suite for Person Re-Identification.
Computes CMC (Rank-1, Rank-5, Rank-10), mAP (mean Average Precision),
Precision, Recall, F1, FMR (False Match Rate), FNMR (False Non-Match Rate),
Inference Latency, Throughput (FPS), GPU memory, and CPU memory usage.
Calculated empirically from actual experimental matrices.
"""

import time
import numpy as np
import torch
import psutil
from typing import Dict, Any, List, Tuple


def evaluate_rank_and_map(distmat: np.ndarray,
                          q_pids: np.ndarray,
                          g_pids: np.ndarray,
                          q_camids: np.ndarray,
                          g_camids: np.ndarray,
                          max_rank: int = 50) -> Tuple[np.ndarray, float]:
    """
    Standard Re-ID Cumulative Matching Characteristic (CMC) and mAP calculation.
    Filters out junk matches (same PID and same Camera ID) as per official benchmark rules.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_cam = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_cam)
        keep = np.invert(remove)

        # Compute CMC
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1

        # Compute AP
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.array(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    if num_valid_q == 0:
        return np.zeros(max_rank), 0.0

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = float(np.mean(all_AP))

    return all_cmc, mAP


def compute_biometric_verification_metrics(similarity_scores: np.ndarray,
                                          labels: np.ndarray,
                                          threshold: float = 0.65) -> Dict[str, float]:
    """
    Computes Precision, Recall, F1, FMR (False Match Rate / FPR), and FNMR (False Non-Match Rate / FNR).
    """
    preds = (similarity_scores >= threshold).astype(int)
    y_true = labels.astype(int)

    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    tn = np.sum((preds == 0) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))

    precision = float(tp / (tp + fp + 1e-12))
    recall = float(tp / (tp + fn + 1e-12))
    f1 = float(2 * precision * recall / (precision + recall + 1e-12))

    fmr = float(fp / (fp + tn + 1e-12)) # False Match Rate (False Positive Rate)
    fnmr = float(fn / (fn + tp + 1e-12)) # False Non-Match Rate (False Negative Rate)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "fmr": fmr,
        "fnmr": fnmr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


class SystemResourceProfiler:
    """
    Measures runtime hardware resource consumption:
    Latency (ms/image), Throughput (FPS), GPU memory (MB), and CPU memory (MB).
    """
    @staticmethod
    def measure_model_efficiency(model_func, sample_input, num_warmup: int = None, num_runs: int = None) -> Dict[str, float]:
        if num_warmup is None:
            num_warmup = 5 if torch.cuda.is_available() else 2
        if num_runs is None:
            num_runs = 20 if torch.cuda.is_available() else 5
        # Measure CPU Memory before
        process = psutil.Process()
        cpu_mem_before = process.memory_info().rss / (1024 * 1024)

        gpu_mem_peak = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Warmup
        for _ in range(num_warmup):
            _ = model_func(sample_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark runs
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model_func(sample_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

        total_time = time.perf_counter() - start_time
        batch_size = sample_input.size(0) if isinstance(sample_input, torch.Tensor) else 1
        total_images = num_runs * batch_size

        latency_ms = (total_time / total_images) * 1000.0
        fps = total_images / total_time
        cpu_mem_after = process.memory_info().rss / (1024 * 1024)

        return {
            "latency_ms": float(round(latency_ms, 2)),
            "fps": float(round(fps, 1)),
            "gpu_memory_mb": float(round(gpu_mem_peak, 1)),
            "cpu_memory_mb": float(round(cpu_mem_after - cpu_mem_before, 1))
        }
