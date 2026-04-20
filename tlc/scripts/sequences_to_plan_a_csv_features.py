#!/usr/bin/env python3
"""
从 *_{train,valid,test}_sequences.json 导出 Plan A（KermtFinetune）用的扁平 CSV + 21 维 npz。

每行对应一个 (SMILES, 溶剂条件) 样本；特征为 solvent(5) + cross(10) + desc(6)，与 build_features / build_sequences 一致。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def export_one(seq_path: Path, csv_path: Path, npz_path: Path) -> None:
    data = json.loads(seq_path.read_text(encoding="utf-8"))
    data.sort(key=lambda x: x["smiles"])
    rows: list[tuple[str, str]] = []
    feats: list[list[float]] = []
    for mol in data:
        smi = mol["smiles"]
        for st in mol["steps"]:
            rf = float(st["rf"])
            rf = max(0.0, min(1.0, rf))
            vec = st["solvent"] + st["cross"] + st["desc"]
            if len(vec) != 21:
                raise ValueError(f"Expected 21-dim features, got {len(vec)} at {smi}")
            rows.append((smi, f"{rf:.6f}"))
            feats.append(vec)
    arr = np.asarray(feats, dtype=np.float32)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "Rf"])
        w.writerows(rows)
    np.savez_compressed(npz_path, features=arr)
    print(f"{seq_path.name} -> {len(rows)} rows, features {arr.shape}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "data_v2_cleaned",
        help="Directory containing *_sequences.json",
    )
    args = p.parse_args()
    base: Path = args.dir
    for split in ("train", "valid", "test"):
        export_one(
            base / f"{split}_sequences.json",
            base / f"{split}.csv",
            base / f"{split}_features.npz",
        )


if __name__ == "__main__":
    main()
