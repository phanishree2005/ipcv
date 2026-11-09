#!/usr/bin/env python3
"""
M²-ReID Adaptive Fusion Network Trainer Script.
Trains the multi-model gating network on Market-1501 identities using Triplet Loss,
Label-Smoothing Cross-Entropy, and Gating Entropy Regularization.
"""

import sys
import argparse
from pathlib import Path
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import set_seed
from src.data.market1501_loader import Market1501DataLoader
from src.reid.model_factory import build_reid_model
from src.fusion.adaptive_fusion import AdaptiveMultiModelFusion
from src.fusion.trainer import AdaptiveFusionTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train M²-ReID Adaptive Fusion Network")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches per epoch for quick training")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    device = "cuda" if (torch.cuda.is_available() and args.device in ["auto", "cuda"]) else "cpu"
    print(f"\n=======================================================")
    print(f"       M²-ReID ADAPTIVE FUSION TRAINING")
    print(f"       Device: {device.upper()} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f"=======================================================\n")

    # Load dataset configuration
    with open(PROJECT_ROOT / "configs" / "dataset.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)

    market_root = PROJECT_ROOT / data_cfg["datasets"]["market1501"]["root"]
    if not market_root.exists():
        print(f"[ERROR] Market-1501 not found at {market_root}")
        return

    # Data Loader
    loader_builder = Market1501DataLoader(
        root_dir=str(market_root),
        batch_size=args.batch_size,
        num_workers=0
    )
    loaders = loader_builder.get_dataloaders()
    train_loader = loaders["train_loader"]
    num_classes = loaders["num_train_pids"]

    print(f"[DATA] Market-1501 Training identities: {num_classes}, Batches per epoch: {len(train_loader)}")

    # Models
    print("[MODELS] Loading frozen feature extractors (OSNet, TransReID, CLIP)...")
    extractor_osnet = build_reid_model("osnet", num_classes=num_classes, device=device).eval()
    extractor_trans = build_reid_model("transreid", num_classes=num_classes, device=device).eval()
    extractor_clip = build_reid_model("clip", num_classes=num_classes, device=device).eval()

    print("[MODELS] Initializing Adaptive Multi-Model Fusion Network...")
    fusion_model = AdaptiveMultiModelFusion(num_classes=num_classes).to(device)

    trainer = AdaptiveFusionTrainer(
        fusion_model=fusion_model,
        num_classes=num_classes,
        lr=args.lr,
        device=device,
        checkpoint_dir=str(PROJECT_ROOT / "checkpoints")
    )

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        metrics = trainer.train_epoch(
            train_loader=train_loader,
            extractor_osnet=extractor_osnet,
            extractor_trans=extractor_trans,
            extractor_clip=extractor_clip,
            epoch=epoch,
            max_batches=args.max_batches
        )
        print(f"Epoch {epoch:02d} Summary: Total Loss: {metrics['loss']:.4f} | "
              f"CE Loss: {metrics['loss_ce']:.4f} | Triplet Loss: {metrics['loss_triplet']:.4f} | "
              f"LR: {metrics['lr']:.6f}")

    trainer.save_checkpoint("adaptive_fusion_market1501.pth")
    print(f"\n[DONE] Adaptive Fusion training complete! Checkpoint saved.")


if __name__ == "__main__":
    main()
