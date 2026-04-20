"""
将 TLC Rf 数据按分子分组、按溶剂体系排序，生成自回归训练所需的序列数据。

输出格式 (JSON):
[
  {
    "smiles": "CCO...",
    "steps": [
      {"solvent": [0.95, 0.05, 0, 0, 0], "cross": [...10d...], "desc": [...6d...], "rf": 0.10},
      {"solvent": [0.90, 0.10, 0, 0, 0], "cross": [...10d...], "desc": [...6d...], "rf": 0.20},
      ...
    ],
    "system_boundaries": [0, 3, 5]  # 体系切换位置 (step index)
  },
  ...
]

溶剂体系识别规则:
  - PE/EA:    H + EA ≈ 1
  - DCM/MeOH: DCM + MeOH ≈ 1
  - PE/Et2O:  H + Et2O ≈ 1
  - 其他组合按主成分归类
"""
import csv
import json
import os
import sys
from collections import defaultdict
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
CROSS_PAIRS = list(combinations(range(5), 2))

SPLITS = {
    "train_data.csv": "train_sequences.json",
    "valid_data.csv": "valid_sequences.json",
    "test_data.csv": "test_sequences.json",
}


def identify_system(solvent):
    """Identify solvent system from 5-dim vector [H, EA, DCM, MeOH, Et2O]."""
    h, ea, dcm, meoh, et2o = solvent
    tol = 0.05
    if dcm + meoh > tol and h + ea + et2o < tol:
        return "DCM_MeOH", meoh / (dcm + meoh + 1e-9)
    if h + et2o > tol and ea + dcm + meoh < tol:
        return "PE_Et2O", et2o / (h + et2o + 1e-9)
    if h + ea > tol and dcm + meoh + et2o < tol:
        return "PE_EA", ea / (h + ea + 1e-9)
    top2 = sorted(enumerate(solvent), key=lambda x: -x[1])[:2]
    return f"OTHER_{top2[0][0]}_{top2[1][0]}", solvent[top2[1][0]] / (sum(solvent) + 1e-9)


def process_split(src_path, dst_path, desc_mean=None, desc_std=None):
    mol_groups = defaultdict(list)

    with open(src_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row["COMPOUND_SMILES"].strip()
            if not smi:
                continue
            solvent = [float(row.get(c, 0) or 0) for c in SOLVENT_COLS]
            cross = [solvent[i] * solvent[j] for i, j in CROSS_PAIRS]
            desc = [float(row.get(c, 0) or 0) for c in DESC_COLS]
            rf = max(0.0, min(1.0, float(row.get("Rf", 0))))

            mol_groups[smi].append({
                "solvent": solvent,
                "cross": cross,
                "desc_raw": desc,
                "rf": rf,
            })

    if desc_mean is None:
        all_desc = []
        for steps in mol_groups.values():
            for s in steps:
                all_desc.append(s["desc_raw"])
        all_desc = np.array(all_desc, dtype=np.float32)
        desc_mean = all_desc.mean(axis=0)
        desc_std = all_desc.std(axis=0)
        desc_std[desc_std < 1e-8] = 1.0

    sequences = []
    total_steps = 0
    system_counts = defaultdict(int)

    for smi, entries in mol_groups.items():
        for e in entries:
            d = np.array(e["desc_raw"], dtype=np.float32)
            e["desc"] = ((d - desc_mean) / desc_std).tolist()

        tagged = []
        for e in entries:
            sys_name, polarity = identify_system(e["solvent"])
            tagged.append((sys_name, polarity, e))

        tagged.sort(key=lambda x: (x[0], x[1]))

        steps = []
        system_boundaries = [0]
        prev_sys = None
        for sys_name, polarity, e in tagged:
            system_counts[sys_name] += 1
            if prev_sys is not None and sys_name != prev_sys:
                system_boundaries.append(len(steps))
            prev_sys = sys_name
            steps.append({
                "solvent": e["solvent"],
                "cross": e["cross"],
                "desc": e["desc"],
                "rf": e["rf"],
                "system": sys_name,
            })

        if len(steps) > 0:
            sequences.append({
                "smiles": smi,
                "steps": steps,
                "system_boundaries": system_boundaries,
                "n_steps": len(steps),
            })
            total_steps += len(steps)

    sequences.sort(key=lambda x: -x["n_steps"])

    with open(dst_path, "w") as f:
        json.dump(sequences, f, ensure_ascii=False)

    seq_lens = [s["n_steps"] for s in sequences]
    print(f"  Molecules: {len(sequences)}")
    print(f"  Total steps: {total_steps}")
    print(f"  Seq len: min={min(seq_lens)}, max={max(seq_lens)}, "
          f"mean={np.mean(seq_lens):.1f}, median={np.median(seq_lens):.0f}")
    print(f"  Systems: {dict(system_counts)}")

    return desc_mean, desc_std


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    desc_mean, desc_std = None, None
    for src_name, dst_name in SPLITS.items():
        src_path = os.path.join(SRC_DIR, src_name)
        dst_path = os.path.join(DST_DIR, dst_name)
        print(f"\n{src_name} → {dst_name}:")
        desc_mean, desc_std = process_split(
            src_path, dst_path,
            desc_mean=desc_mean if "train" not in src_name else None,
            desc_std=desc_std if "train" not in src_name else None,
        )

    np.savez(
        os.path.join(DST_DIR, "desc_stats.npz"),
        mean=desc_mean, std=desc_std,
    )
    print(f"\nDesc normalization stats saved to {DST_DIR}/desc_stats.npz")


if __name__ == "__main__":
    main()
