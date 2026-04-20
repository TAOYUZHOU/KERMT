#!/usr/bin/env python3
"""
从与 build_features.py 相同的 train_data.csv 计算描述符 6 维的 mean/std（z-score 用），
写入 JSON，供单分子 predict_smiles_solvent 使用。

默认 SRC_DATA_DIR 与 build_features.py 一致。

用法:
  python tlc/scripts/export_descriptor_zscore_stats.py
  SRC_DATA_DIR=/path/to/unimol_split python tlc/scripts/export_descriptor_zscore_stats.py --out tlc/data/descriptor_zscore_stats.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np

SOLVENT_COLS = ["H", "EA", "DCM", "MeOH", "Et2O"]
DESC_COLS = ["MW", "TPSA", "LogP", "HBD", "HBA", "NROTBs"]
CROSS_PAIRS = list(combinations(SOLVENT_COLS, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default=os.environ.get(
            "SRC_DATA_DIR",
            "/root/autodl-tmp/taoyuzhou/unimol_tlc/merged_data_molecule_split_v2_rf_stratified_heavy70_descriptor",
        ),
        help="含 train_data.csv 的目录",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "descriptor_zscore_stats.json",
    )
    args = ap.parse_args()

    src_path = Path(args.src) / "train_data.csv"
    if not src_path.is_file():
        raise SystemExit(f"Missing {src_path}")

    rows = []
    with open(src_path) as f:
        for row in csv.DictReader(f):
            smi = row.get("COMPOUND_SMILES", "").strip()
            if not smi:
                continue
            solvent = [float(row.get(c, 0) or 0) for c in SOLVENT_COLS]
            cross = [
                solvent[SOLVENT_COLS.index(a)] * solvent[SOLVENT_COLS.index(b)]
                for a, b in CROSS_PAIRS
            ]
            desc = [float(row.get(c, 0) or 0) for c in DESC_COLS]
            rows.append(solvent + cross + desc)

    features = np.array(rows, dtype=np.float64)
    desc_start = len(SOLVENT_COLS) + len(CROSS_PAIRS)
    block = features[:, desc_start:]
    mean = block.mean(axis=0)
    std = block.std(axis=0)
    std[std < 1e-8] = 1.0

    payload = {
        "desc_mean": mean.tolist(),
        "desc_std": std.tolist(),
        "solvent_cols": SOLVENT_COLS,
        "desc_cols": DESC_COLS,
        "cross_pairs": [[a, b] for a, b in CROSS_PAIRS],
        "n_train_rows": len(rows),
        "source_train_csv": str(src_path.resolve()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} (n={len(rows)})")


if __name__ == "__main__":
    main()
