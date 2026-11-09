"""
Logging and reproducibility utilities for M²-ReID.
Saves experiment runs with hardware, software versions, configurations, random seeds, and metrics.
"""

import os
import sys
import json
import time
import platform
import random
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """Sets random seeds across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_environment_info() -> Dict[str, Any]:
    """Captures runtime hardware and software environment metadata."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cpu_count": os.cpu_count()
    }


def save_experiment_result(experiment_id: str,
                           dataset: str,
                           model_name: str,
                           config: Dict[str, Any],
                           metrics: Dict[str, Any],
                           output_dir: str = "results") -> str:
    """
    Saves structured JSON record of experiment for reproducibility.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{experiment_id}.json"

    # Sanitize metrics: exclude large distance matrices
    clean_metrics = {}
    for k, v in metrics.items():
        if k == "distmat":
            continue
        if isinstance(v, np.ndarray):
            clean_metrics[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            clean_metrics[k] = float(v)
        else:
            clean_metrics[k] = v

    record = {
        "experiment_id": experiment_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "model": model_name,
        "random_seed": config.get("random_seed", 42),
        "configuration": config,
        "metrics": clean_metrics,
        "environment": get_environment_info()
    }

    with open(file_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"[REPRODUCIBILITY] Saved experiment record to {file_path}")
    return str(file_path)
