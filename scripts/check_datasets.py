#!/usr/bin/env python3
"""
M²-ReID Dataset Check and Download Guidance Assistant
Checks availability and integrity of Market-1501 and MSMT17 datasets.
Adheres strictly to responsible research dataset licensing.
"""

import os
import sys
from pathlib import Path
import yaml

def load_dataset_config():
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "configs" / "dataset.yaml"
    if not config_path.exists():
        print(f"[ERROR] Configuration not found at: {config_path}")
        return None, base_dir
    with open(config_path, "r") as f:
        return yaml.safe_load(f), base_dir

def check_market1501(cfg, base_dir):
    print("=" * 70)
    print("1. CHECKING DATASET 1: Market-1501 (Development / Baseline Benchmark)")
    print("=" * 70)
    
    market_cfg = cfg.get("datasets", {}).get("market1501", {})
    root_rel = market_cfg.get("root", "data/market1501/Market-1501-v15.09.15")
    root_path = base_dir / root_rel
    
    required_dirs = [
        market_cfg.get("train_dir", "bounding_box_train"),
        market_cfg.get("query_dir", "query"),
        market_cfg.get("gallery_dir", "bounding_box_test"),
    ]
    
    exists = root_path.exists()
    all_subdirs_exist = exists and all((root_path / d).exists() for d in required_dirs)
    
    if all_subdirs_exist:
        train_count = len(list((root_path / required_dirs[0]).glob("*.jpg")))
        query_count = len(list((root_path / required_dirs[1]).glob("*.jpg")))
        gallery_count = len(list((root_path / required_dirs[2]).glob("*.jpg")))
        
        print(f"Market-1501: [FOUND]")
        print(f"  Location: {root_path}")
        print(f"  Train images ({required_dirs[0]}): {train_count}")
        print(f"  Query images ({required_dirs[1]}): {query_count}")
        print(f"  Gallery images ({required_dirs[2]}): {gallery_count}")
        return True
    else:
        print("Market-1501: [NOT FOUND]")
        print(f"  Expected location: {root_path}")
        print("\n--- INSTRUCTIONS TO OBTAIN MARKET-1501 ---")
        print("Market-1501 contains 1,501 identities captured across 6 camera views.")
        print("Legitimate Research Repository:")
        print(f"  Source: {market_cfg.get('source_url', 'https://github.com/doosik71/market-1501')}")
        print("Archives needed:")
        print("  - Market-1501-v15.09.15-train.zip (bounding_box_train)")
        print("  - Market-1501-v15.09.15-query.zip (query)")
        print("  - Market-1501-v15.09.15-test.zip (bounding_box_test)")
        print(f"\nPlace and extract into: {root_path.parent}")
        print("-------------------------------------------\n")
        return False

def check_msmt17(cfg, base_dir):
    print("=" * 70)
    print("2. CHECKING DATASET 2: MSMT17 (Challenging Generalization Benchmark)")
    print("=" * 70)
    
    msmt_cfg = cfg.get("datasets", {}).get("msmt17", {})
    root_rel = msmt_cfg.get("root", "data/msmt17/MSMT17_V1")
    root_path = base_dir / root_rel
    
    required_dirs = [
        msmt_cfg.get("train_dir", "train"),
        msmt_cfg.get("test_dir", "test"),
    ]
    required_files = [
        msmt_cfg.get("list_train", "list_train.txt"),
        msmt_cfg.get("list_val", "list_val.txt"),
        msmt_cfg.get("list_query", "list_query.txt"),
        msmt_cfg.get("list_gallery", "list_gallery.txt"),
    ]
    
    exists = root_path.exists()
    dirs_exist = exists and all((root_path / d).exists() for d in required_dirs)
    files_exist = exists and all((root_path / f).exists() for f in required_files)
    
    if dirs_exist and files_exist:
        train_count = len(list((root_path / required_dirs[0]).rglob("*.jpg")))
        test_count = len(list((root_path / required_dirs[1]).rglob("*.jpg")))
        print("MSMT17: [FOUND]")
        print(f"  Location: {root_path}")
        print(f"  Train images: {train_count}")
        print(f"  Test images: {test_count}")
        return True
    else:
        print("MSMT17: [NOT FOUND]")
        print(f"  Expected location: {root_path}")
        print("\n--- OFFICIAL MSMT17 DATASET ACCESS INSTRUCTIONS ---")
        print("MSMT17 (Multi-Scene Multi-Time) contains 4,101 IDs across 15 cameras.")
        print(f"Official Portal: {msmt_cfg.get('official_url', 'https://www.pkuvmc.com/dataset/dataset.html')}")
        print("IMPORTANT REQUIREMENT:")
        print("  MSMT17 requires an approved dataset agreement before obtaining download credentials.")
        print(f"  Submit your signed agreement to: {msmt_cfg.get('official_contact', 'slzhang.jdl@pku.edu.cn')}")
        print("  Do NOT attempt to bypass this requirement or download from unauthorized mirrors.")
        print(f"\nAfter approval, extract the official archive into: {root_path}")
        print("----------------------------------------------------\n")
        return False

def main():
    cfg, base_dir = load_dataset_config()
    if not cfg:
        sys.exit(1)
        
    print("\n" + "#" * 70)
    print("      M²-ReID RESEARCH DATASET VALIDATION & ASSISTANT")
    print("#" * 70 + "\n")
    
    m_found = check_market1501(cfg, base_dir)
    print()
    ms_found = check_msmt17(cfg, base_dir)
    
    print("=" * 70)
    print("DATASET STATUS SUMMARY:")
    print(f"  Market-1501: [{'FOUND' if m_found else 'NOT FOUND'}]")
    print(f"  MSMT17:      [{'FOUND' if ms_found else 'NOT FOUND'}]")
    print("=" * 70)
    
    if not m_found or not ms_found:
        print("\n[NOTE] You can also run 'python scripts/generate_demo_data.py' to create")
        print("a mini synthetic verification dataset for quick pipeline sanity tests.")
    print()

if __name__ == "__main__":
    main()
