"""
从原始 UniMol TLC 数据生成 KERMT features npz 文件。

特征向量 [N, 21]:
  - 溶剂比例:    H, EA, DCM, MeOH, Et2O           (5 维)
  - 溶剂交叉项:  H×EA, H×DCM, ..., MeOH×Et2O      (10 维)
  - 分子描述符:   MW, TPSA, LogP, HBD, HBA, NROTBs (6 维)

用法:
  python tlc/scripts/build_features.py
"""
import csv
import os
import sys
from itertools import combinations

import numpy as np

SRC_DIR = os.environ.get(
    "SRC_DATA_DIR",
    "/root/autodl-tmp/taoyuzhou/unimol_tlc/merged_data_molecule_split_v2_rf_stratified_heavy70_descriptor",
)
DST_DIR = os.environ.get(
    "DST_DATA_DIR",
    "/root/autodl-tmp/taoyuzhou/KERMT/tlc/data",
)

SOLVENT_COLS = ["H", "EA", "DCM", "MeOH", "Et2O"]
DESC_COLS = ["MW", "TPSA", "LogP", "HBD", "HBA", "NROTBs"]
CROSS_PAIRS = list(combinations(SOLVENT_COLS, 2))

SPLITS = {
    "train_data.csv": "train_features.npz",
    "valid_data.csv": "valid_features.npz",
    "test_data.csv": "test_features.npz",
}

os.makedirs(DST_DIR, exist_ok=True)

desc_stats = None

for split_src, split_dst in SPLITS.items():
    src_path = os.path.join(SRC_DIR, split_src)
    dst_path = os.path.join(DST_DIR, split_dst)

    rows = []
    with open(src_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row["COMPOUND_SMILES"].strip()
            if not smi:
                continue

            solvent = [float(row.get(c, 0) or 0) for c in SOLVENT_COLS]
            cross = [solvent[i] * solvent[j] for i, j in
                     [(SOLVENT_COLS.index(a), SOLVENT_COLS.index(b)) for a, b in CROSS_PAIRS]]
            desc = [float(row.get(c, 0) or 0) for c in DESC_COLS]

            rows.append(solvent + cross + desc)

    features = np.array(rows, dtype=np.float32)

    if desc_stats is None:
        desc_start = len(SOLVENT_COLS) + len(CROSS_PAIRS)
        desc_block = features[:, desc_start:]
        desc_stats = {
            "mean": desc_block.mean(axis=0),
            "std": desc_block.std(axis=0),
        }
        desc_stats["std"][desc_stats["std"] < 1e-8] = 1.0

    desc_start = len(SOLVENT_COLS) + len(CROSS_PAIRS)
    features[:, desc_start:] = (
        (features[:, desc_start:] - desc_stats["mean"]) / desc_stats["std"]
    )

    np.savez_compressed(dst_path, features=features)

    dim_label = (
        f"solvent({len(SOLVENT_COLS)}) + "
        f"cross({len(CROSS_PAIRS)}) + "
        f"desc({len(DESC_COLS)})"
    )
    print(f"{split_src:20s} → {split_dst:24s}  shape={features.shape}  [{dim_label}]")

print(f"\nFeature dim: {features.shape[1]}")
print(f"Solvent cols: {SOLVENT_COLS}")
print(f"Cross pairs:  {[f'{a}×{b}' for a, b in CROSS_PAIRS]}")
print(f"Desc cols:    {DESC_COLS} (z-score normalized from train)")
print(f"Output: {DST_DIR}")
