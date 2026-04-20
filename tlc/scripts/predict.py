#!/usr/bin/env python
"""
KERMT TLC 预测 + 评估入口 — 从 config.yaml 或命令行读取参数

用法:
  # 方式 1: 使用训练时自动保存的 config（推荐）
  python tlc/scripts/predict.py --checkpoint_dir tlc/results/default/fold_0/model_0

  # 方式 2: 指定 config + 覆盖
  python tlc/scripts/predict.py --config tlc/configs/default.yaml \
      --checkpoint_dir tlc/results/default/fold_0/model_0 \
      --data_path tlc/data/test.csv
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import yaml
from rdkit import RDLogger

KERMT_ROOT = str(Path(__file__).resolve().parents[2])
if KERMT_ROOT not in sys.path:
    sys.path.insert(0, KERMT_ROOT)

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}

from kermt.util.parsing import get_newest_train_args
from kermt.data.torchvocab import MolVocab
from task.predict import make_predictions, write_prediction


def evaluate(pred_csv: str, true_csv: str):
    with open(true_csv) as f:
        reader = csv.DictReader(f)
        y_true = [float(r.get("Rf", r.get("y_mean", 0))) for r in reader]

    with open(pred_csv) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        y_pred = [float(r[cols[-1]]) for r in reader]

    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)
    mean_t = sum(y_true) / n
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    ss_tot = sum((t - mean_t) ** 2 for t in y_true)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print("=" * 48)
    print(f"  Evaluation Results (n={n})")
    print("=" * 48)
    print(f"  MAE:   {mae:.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  R²:    {r2:.4f}")
    print(f"  True range:  [{min(y_true):.4f}, {max(y_true):.4f}]")
    print(f"  Pred range:  [{min(y_pred):.4f}, {max(y_pred):.4f}]")
    print("=" * 48)

    return {"mae": mae, "rmse": rmse, "r2": r2, "n": n}


def main():
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    _ = MolVocab

    cli_parser = argparse.ArgumentParser(add_help=False)
    cli_parser.add_argument("--config", type=str, default=None)
    cli_parser.add_argument("--checkpoint_dir", type=str, required=True)
    cli_parser.add_argument("--data_path", type=str, default=None)
    cli_parser.add_argument("--output", type=str, default=None)
    cli_parser.add_argument("--no_eval", action="store_true", default=False)
    cli_known, cli_rest = cli_parser.parse_known_args()

    ckpt_dir = cli_known.checkpoint_dir
    if not os.path.isabs(ckpt_dir):
        ckpt_dir = os.path.join(KERMT_ROOT, ckpt_dir)

    config_path = cli_known.config
    if config_path is None:
        auto_cfg = os.path.join(os.path.dirname(os.path.dirname(ckpt_dir)), "config.yaml")
        if os.path.isfile(auto_cfg):
            config_path = auto_cfg
            print(f"Auto-detected config: {config_path}")

    if config_path:
        if not os.path.isabs(config_path):
            config_path = os.path.join(KERMT_ROOT, config_path)
        cfg = load_yaml(config_path)
    else:
        cfg = {}

    test_csv = cli_known.data_path
    if test_csv is None:
        test_csv = cfg.get("separate_test_path", "tlc/data/test.csv")
    if not os.path.isabs(test_csv):
        test_csv = os.path.join(KERMT_ROOT, test_csv)

    output_csv = cli_known.output
    if output_csv is None:
        output_csv = os.path.join(ckpt_dir, "predictions.csv")
    elif not os.path.isabs(output_csv):
        output_csv = os.path.join(KERMT_ROOT, output_csv)

    predict_overrides = [
        "--data_path", test_csv,
        "--checkpoint_dir", ckpt_dir,
        "--output_path", output_csv,
    ]

    # CLI 显式传入 --features_path 时（例如按 split 重跑 train/valid/test），不再用 config 里的 test 特征
    use_cli_features = "--features_path" in cli_rest

    features_path = None if use_cli_features else cfg.get(
        "separate_test_features_path", cfg.get("features_path")
    )
    if features_path:
        if isinstance(features_path, list):
            predict_overrides.append("--features_path")
            for fp in features_path:
                if not os.path.isabs(fp):
                    fp = os.path.join(KERMT_ROOT, fp)
                predict_overrides.append(fp)
        elif isinstance(features_path, str):
            fp = features_path
            if not os.path.isabs(fp):
                fp = os.path.join(KERMT_ROOT, fp)
            predict_overrides.extend(["--features_path", fp])

    if cfg.get("no_features_scaling"):
        predict_overrides.append("--no_features_scaling")
    if cfg.get("no_cuda", True):
        predict_overrides.append("--no_cuda")

    predict_overrides.extend(cli_rest)

    from kermt.util.parsing import (
        add_predict_args,
        modify_predict_args,
    )
    pred_parser = argparse.ArgumentParser()
    add_predict_args(pred_parser)
    args = pred_parser.parse_args(predict_overrides)
    args.parser_name = "predict"
    modify_predict_args(args)

    args.output = output_csv

    print("=" * 56)
    print("  KERMT TLC Prediction (config-driven)")
    print("=" * 56)
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"  Test data:  {test_csv}")
    print(f"  Output:     {output_csv}")
    print("=" * 56)

    train_args = get_newest_train_args()
    avg_preds, test_smiles = make_predictions(args, train_args)
    write_prediction(avg_preds, test_smiles, args)

    print(f"\nPredictions saved to: {output_csv}")

    if not cli_known.no_eval:
        evaluate(output_csv, test_csv)


if __name__ == "__main__":
    main()
