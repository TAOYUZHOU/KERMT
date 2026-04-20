"""
将 UniMol TLC Rf 数据转换为 KERMT 所需的 CSV 格式。

UniMol 格式: COMPOUND_SMILES, H, EA, DCM, MeOH, Et2O, Rf, ...descriptors
KERMT 格式:  smiles, Rf

KERMT 不支持溶剂条件作为输入特征（纯图模型），
因此每条 (SMILES + 溶剂组合 → Rf) 都作为独立样本。
后续可通过 --features_path 传入溶剂/描述符作为额外特征。
"""
import csv
import os
import sys

SRC_DIR = os.environ.get(
    "SRC_DATA_DIR",
    "/root/autodl-tmp/taoyuzhou/unimol_tlc/merged_data_molecule_split_v2_rf_stratified_heavy70_descriptor",
)
DST_DIR = os.environ.get(
    "DST_DATA_DIR",
    "/root/autodl-tmp/taoyuzhou/KERMT/tlc/data",
)

SOLVENT_COLS = ["H", "EA", "DCM", "MeOH", "Et2O"]
DESC_COLS = ["MW", "C_COUNT", "HEAVY_ATOMS", "TPSA", "LogP", "HBA", "HBD", "NROTBs"]

os.makedirs(DST_DIR, exist_ok=True)

for split in ("train_data.csv", "valid_data.csv", "test_data.csv"):
    src_path = os.path.join(SRC_DIR, split)
    out_name = split.replace("_data", "")
    dst_path = os.path.join(DST_DIR, out_name)

    with open(src_path) as fin, open(dst_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["smiles", "Rf"])

        count = 0
        skipped = 0
        for row in reader:
            smi = row["COMPOUND_SMILES"].strip()
            rf_str = row["Rf"].strip()
            if not smi or not rf_str:
                skipped += 1
                continue
            rf = float(rf_str)
            rf = max(0.0, min(1.0, rf))
            writer.writerow([smi, f"{rf:.6f}"])
            count += 1

        print(f"{split:20s} → {out_name:12s}  rows={count}  skipped={skipped}")

print(f"\nOutput: {DST_DIR}")
