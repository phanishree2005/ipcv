#!/usr/bin/env python3
"""
M²-ReID Robustness and Perturbation Resilience Experiment.
Evaluates model degradation curves under:
1. Gaussian Blur (levels 1, 2, 3, 4)
2. Extreme Brightness shifts (0.3x dark to 1.7x bright)
3. Contrast degradation (0.3x, 0.6x)
4. Synthetic Occlusions (10%, 25%, 40% area)
5. Low Resolution downsampling (2x, 4x, 8x)
Compares OSNet Baseline vs Adaptive Multi-Model M²-ReID.
"""

import sys
import json
from pathlib import Path
import numpy as np
import cv2
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import set_seed
from src.data.market1501_loader import Market1501DataLoader
from src.reid.model_factory import build_reid_model
from src.reid.base_extractor import normalize_features
from src.fusion.adaptive_fusion import AdaptiveMultiModelFusion
from src.evaluation.evaluator import ReIDEvaluator
from src.evaluation.robustness import RobustnessEvaluator
from src.evaluation.metrics import evaluate_rank_and_map
from src.data.transforms import build_reid_transforms
from src.utils.visualization import plot_robustness_curves

def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n=======================================================")
    print("      M²-ReID ROBUSTNESS & PERTURBATION EXPERIMENTS")
    print("=======================================================\n")

    with open(PROJECT_ROOT / "configs" / "dataset.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)

    market_root = PROJECT_ROOT / data_cfg["datasets"]["market1501"]["root"]
    if not market_root.exists():
        from scripts.generate_demo_data import generate_market1501_demo
        generate_market1501_demo(PROJECT_ROOT)

    loader_builder = Market1501DataLoader(str(market_root), batch_size=16, num_workers=0)
    loaders = loader_builder.get_dataloaders()
    query_loader = loaders["query_loader"]
    gallery_loader = loaders["gallery_loader"]
    num_classes = loaders["num_train_pids"]

    evaluator = ReIDEvaluator(device=device)
    model_osnet = build_reid_model("osnet", num_classes=num_classes, device=device).eval()
    model_trans = build_reid_model("transreid", num_classes=num_classes, device=device).eval()
    model_clip = build_reid_model("clip", num_classes=num_classes, device=device).eval()
    adaptive_model = AdaptiveMultiModelFusion(num_classes=num_classes).to(device).eval()

    # Base unperturbed gallery features
    g_os, g_pids, g_cams, g_paths = evaluator.extract_features(model_osnet, gallery_loader)
    g_tr, _, _, _ = evaluator.extract_features(model_trans, gallery_loader)
    g_cl, _, _, _ = evaluator.extract_features(model_clip, gallery_loader)

    with torch.no_grad():
        g_ad, _ = adaptive_model(
            torch.from_numpy(g_os).to(device),
            torch.from_numpy(g_tr).to(device),
            torch.from_numpy(g_cl).to(device)
        )
    g_adapt = g_ad.cpu().numpy()

    test_transform = build_reid_transforms(256, 128, is_train=False)

    perturbations = [
        ("blur", 1, "Blur (Level 1, k=3)"),
        ("blur", 2, "Blur (Level 2, k=5)"),
        ("blur", 3, "Blur (Level 3, k=7)"),
        ("blur", 4, "Blur (Level 4, k=9)"),
        ("brightness_low", 0.4, "Low Brightness (0.4x)"),
        ("brightness_high", 1.6, "High Brightness (1.6x)"),
        ("contrast", 0.4, "Low Contrast (0.4x)"),
        ("occlusion", 0.10, "Occlusion (10% Area)"),
        ("occlusion", 0.25, "Occlusion (25% Area)"),
        ("occlusion", 0.40, "Occlusion (40% Area)"),
        ("low_res", 2, "Low Resolution (2x Downsample)"),
        ("low_res", 4, "Low Resolution (4x Downsample)"),
    ]

    robustness_results = {}
    print(f"{'Perturbation':<32} | {'OSNet R-1 (%)':<15} | {'Adaptive R-1 (%)':<18} | {'Delta':<8}")
    print("-" * 78)

    query_data = loaders["query_data"]
    q_pids = np.array([item[1] for item in query_data])
    q_cams = np.array([item[2] for item in query_data])

    for p_type, severity, desc in perturbations:
        # Generate perturbed query images
        perturbed_imgs = []
        for img_path, _, _, _ in query_data:
            img = cv2.imread(img_path)
            if img is None:
                img = np.full((256, 128, 3), 128, dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            p_img = RobustnessEvaluator().generate_perturbed_batch([img], p_type, severity)[0]
            t_tensor = test_transform(Image.fromarray(p_img))
            perturbed_imgs.append(t_tensor)

        q_batch = torch.stack(perturbed_imgs, dim=0).to(device)

        with torch.no_grad():
            q_os = model_osnet(q_batch).cpu().numpy()
            q_tr = model_trans(q_batch).cpu().numpy()
            q_cl = model_clip(q_batch).cpu().numpy()
            q_ad, _ = adaptive_model(
                torch.from_numpy(q_os).to(device),
                torch.from_numpy(q_tr).to(device),
                torch.from_numpy(q_cl).to(device)
            )
            q_adapt = q_ad.cpu().numpy()

        res_os = evaluator.evaluate(q_os, g_os, q_pids, g_pids, q_cams, g_cams)
        res_ad = evaluator.evaluate(q_adapt, g_adapt, q_pids, g_pids, q_cams, g_cams)

        delta = res_ad["rank1"] - res_os["rank1"]
        print(f"{desc:<32} | {res_os['rank1']:<15.1f} | {res_ad['rank1']:<18.1f} | {delta:+5.1f}%")

        key = f"{p_type}_{severity}"
        robustness_results[key] = {
            "description": desc,
            "osnet_r1": res_os["rank1"],
            "osnet_map": res_os["mAP"],
            "adapt_r1": res_ad["rank1"],
            "adapt_map": res_ad["mAP"],
            "delta_r1": float(round(delta, 2))
        }

    # Save output and plot
    out_file = PROJECT_ROOT / "results" / "robustness_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(robustness_results, f, indent=2)

    plot_robustness_curves(robustness_results, save_path=str(PROJECT_ROOT / "results" / "plots" / "robustness_curves.png"))
    print("\n[ROBUSTNESS] Benchmark complete. Results saved to results/robustness_benchmark.json and plots/robustness_curves.png\n")

if __name__ == "__main__":
    main()
