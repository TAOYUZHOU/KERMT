#!/usr/bin/env python
"""
Plan A（with_features）评测绘图 — 与 eval_and_plot_autoregressive.py 同一套产物逻辑：

  1) 默认 --rerun：加载 fold_0/model_0，对 train/valid/test 分别调用 predict.py 重跑预测，
     再画 parity_Train/Valid/Test、parity_combined、training_curves（与方案 B 一致）。
  2) --artifact_only：仅读 fold_0/test_result.csv 画一张 Test 散点（不加载模型，仅旧产物）。

predict.py 在命令行传入 --features_path 时会覆盖 config 里默认的 test 特征，
以便对 train/valid 使用各自 npz。

用法:
  cd /root/autodl-tmp/taoyuzhou/KERMT
  python tlc/scripts/eval_and_plot_plan_a.py --result_dir tlc/results/with_features_v1_dirty

  # 仅快速测试集图（不跑 GPU）
  python tlc/scripts/eval_and_plot_plan_a.py --result_dir ... --artifact_only

  # 指定训练日志（用于 loss 曲线；默认同目录下无则跳过）
  python tlc/scripts/eval_and_plot_plan_a.py --result_dir ... --train_log tlc/results/with_features_train.log
"""
import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

KERMT_ROOT = str(Path(__file__).resolve().parents[2])
if KERMT_ROOT not in sys.path:
    sys.path.insert(0, KERMT_ROOT)

SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from tlc_plot_common import (
    plot_combined_parity,
    plot_parity_split,
    plot_training_curves_mse,
)


def _abs(root: str, p: str) -> str:
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(root, p))


def _one_npz(x):
    if x is None:
        return None
    if isinstance(x, list):
        return x[0] if x else None
    return x


def load_test_result_csv(path: str):
    """KERMT test_result.csv → (y_true, y_pred)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        raise ValueError(f"Expected ≥3 rows in {path}")
    y_true, y_pred = [], []
    for row in rows[2:]:
        if len(row) < 3:
            continue
        try:
            y_pred.append(float(row[1]))
            y_true.append(float(row[2]))
        except (ValueError, IndexError):
            continue
    return np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)


def load_truth_pred_from_predict_csv(pred_csv: str, truth_csv: str):
    """与 tlc/scripts/predict.py evaluate() 一致：按行对齐。"""
    with open(truth_csv) as f:
        reader = csv.DictReader(f)
        y_true = [float(r.get("Rf", r.get("y_mean", 0))) for r in reader]
    with open(pred_csv) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        y_pred = [float(r[cols[-1]]) for r in reader]
    n = min(len(y_true), len(y_pred))
    return np.array(y_true[:n], dtype=np.float64), np.array(y_pred[:n], dtype=np.float64)


def run_predict_subprocess(
    ckpt_dir: str,
    config_path: str,
    data_csv: str,
    features_npz: str,
    output_csv: str,
) -> None:
    cmd = [
        sys.executable,
        os.path.join(KERMT_ROOT, "tlc/scripts/predict.py"),
        "--checkpoint_dir",
        ckpt_dir,
        "--config",
        config_path,
        "--data_path",
        data_csv,
        "--output",
        output_csv,
        "--no_eval",
        "--features_path",
        features_npz,
    ]
    print("  ", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = KERMT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call(cmd, cwd=KERMT_ROOT, env=env)


def main():
    p = argparse.ArgumentParser(description="Plan A parity plots (same layout as Plan B)")
    p.add_argument("--result_dir", type=str, required=True,
                   help="训练 save_dir（含 fold_0/model_0 与 config.yaml）")
    p.add_argument("--output_dir", type=str, default=None,
                   help="默认: <result_dir>/plots")
    p.add_argument("--artifact_only", action="store_true",
                   help="只读 fold_0/test_result.csv 画 Test，不重跑模型")
    p.add_argument("--train_log", type=str, default=None,
                   help="Plan A 训练日志（含 Epoch: ... loss_train ... mae_val）；默认尝试自动查找")
    args = p.parse_args()

    rd = args.result_dir
    if not os.path.isabs(rd):
        rd = os.path.join(KERMT_ROOT, rd)
    output_dir = args.output_dir or os.path.join(rd, "plots")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(KERMT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_path = os.path.join(rd, "effective_config.yaml")
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(rd, "config.yaml")

    if args.artifact_only:
        tr = os.path.join(rd, "fold_0", "test_result.csv")
        if not os.path.isfile(tr):
            print(f"找不到 {tr}", file=sys.stderr)
            sys.exit(1)
        y_true, y_pred = load_test_result_csv(tr)
        out_png = os.path.join(output_dir, f"parity_Test_{ts}.png")
        plot_parity_split(
            y_true, y_pred, output_dir, "Test", ts,
            title_prefix="Plan A (artifact test_result.csv)",
            color="#2ca02c",
        )
        np.savetxt(
            os.path.join(output_dir, f"test_predictions_{ts}.csv"),
            np.column_stack([y_true, y_pred]),
            delimiter=",",
            header="true_rf,pred_rf",
            comments="",
        )
        print(f"\nAll outputs under: {output_dir}")
        return

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    plan_title = "Plan A (with_features)"
    if cfg.get("regression_loss") == "beta_nll":
        plan_title = "Plan A (Beta NLL)"

    ckpt_dir = os.path.join(rd, "fold_0", "model_0")
    if not os.path.isdir(ckpt_dir):
        print(f"找不到 checkpoint 目录: {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    train_csv = _abs(KERMT_ROOT, cfg.get("data_path", ""))
    train_npz = _abs(KERMT_ROOT, _one_npz(cfg.get("features_path")))
    val_csv = _abs(KERMT_ROOT, cfg.get("separate_val_path", ""))
    val_npz = _abs(KERMT_ROOT, _one_npz(cfg.get("separate_val_features_path")))
    test_csv = _abs(KERMT_ROOT, cfg.get("separate_test_path", ""))
    test_npz = _abs(KERMT_ROOT, _one_npz(cfg.get("separate_test_features_path")))

    splits = [
        ("Train", train_csv, train_npz),
        ("Valid", val_csv, val_npz),
        ("Test", test_csv, test_npz),
    ]

    print("=" * 60)
    print("  Plan A — rerun predict + parity (same layout as Plan B)")
    print("=" * 60)

    log_path = args.train_log
    if log_path and not os.path.isabs(log_path):
        log_path = _abs(KERMT_ROOT, log_path)
    if not log_path or not os.path.isfile(log_path):
        for cand in [
            os.path.join(rd, "nohup_train.log"),
            os.path.join(rd, "train.log"),
            os.path.join(os.path.dirname(rd), "with_features_train.log"),
            _abs(KERMT_ROOT, "tlc/results/with_features_train.log"),
        ]:
            if os.path.isfile(cand):
                log_path = cand
                break

    if log_path and os.path.isfile(log_path):
        print(f"\n[1/4] Training curves from {log_path}")
        plot_training_curves_mse(log_path, output_dir, ts)
    else:
        print("\n[1/4] No train log found, skip training curves.")

    all_metrics = {}
    step = 2
    for split_name, csv_path, npz_path in splits:
        if not csv_path or not os.path.isfile(csv_path):
            print(f"\n[{step}/4] Skip {split_name}: missing {csv_path}")
            step += 1
            continue
        if not npz_path or not os.path.isfile(npz_path):
            print(f"\n[{step}/4] Skip {split_name}: missing features {npz_path}")
            step += 1
            continue

        print(f"\n[{step}/4] Predict {split_name} …")
        pred_tmp = os.path.join(output_dir, f"_raw_pred_{split_name.lower()}_{ts}.csv")
        run_predict_subprocess(ckpt_dir, cfg_path, csv_path, npz_path, pred_tmp)
        y_true, y_pred = load_truth_pred_from_predict_csv(pred_tmp, csv_path)
        np.savetxt(
            os.path.join(output_dir, f"{split_name.lower()}_predictions.csv"),
            np.column_stack([y_true, y_pred]),
            delimiter=",",
            header="true_rf,pred_rf",
            comments="",
        )
        try:
            os.remove(pred_tmp)
        except OSError:
            pass

        m = plot_parity_split(
            y_true, y_pred, output_dir, split_name, ts,
            title_prefix=plan_title,
            color="#1f77b4",
        )
        all_metrics[split_name] = m
        step += 1

    if len(all_metrics) >= 2:
        plot_combined_parity(
            all_metrics, output_dir, ts,
            suptitle=f"{plan_title} — Parity Plots",
        )

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, m in all_metrics.items():
        print(f"  {name:10s}  R²={m['r2']:.4f}  MAE={m['mae']:.4f}  n={m['n']}")
    print("=" * 60)
    print(f"\nAll outputs under: {output_dir}")


if __name__ == "__main__":
    main()
