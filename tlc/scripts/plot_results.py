"""
Parse KERMT training logs and produce loss curves + prediction scatter plots.

Usage (loss + scatter):
  python tlc/scripts/plot_results.py \\
      --logs tlc/results/with_features_train.log \\
      --labels "Plan A v1 dirty" \\
      --pred_dirs tlc/results/with_features_v1_dirty/fold_0/model_0 \\
      --test_csv tlc/data/test.csv \\
      --output tlc/results/with_features_v1_dirty/plots/plot_results.png

predictions.csv 查找顺序（与 predict.py 默认一致）:
  1) <pred_dir>/predictions.csv   （例如 fold_0/model_0/predictions.csv）
  2) <pred_dir>/../predictions.csv （例如 fold_0/predictions.csv，旧产物）

若无训练 log，可只画测试散点（省略 --logs / --labels）:
  python tlc/scripts/plot_results.py \\
      --pred_dirs tlc/results/with_features_v1_dirty/fold_0/model_0 \\
      --test_csv tlc/data/test.csv \\
      --output tlc/results/with_features_v1_dirty/plots/test_scatter.png
"""
import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
# plot_results.py 位于 KERMT/tlc/scripts/，仓库根为再上两级
_KERMT_ROOT = Path(__file__).resolve().parents[2]

def parse_training_log(log_path: str):
    """Extract epoch, train_loss, val_loss, val_metric from log file."""
    epochs, train_losses, val_losses, val_metrics = [], [], [], []
    pattern = re.compile(
        r"Epoch:\s*(\d+)\s+"
        r"loss_train:\s*([\d.]+)\s+"
        r"loss_val:\s*([\d.]+)\s+"
        r"(\w+)_val:\s*([\d.]+)"
    )
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                val_losses.append(float(m.group(3)))
                val_metrics.append(float(m.group(5)))
    return {
        "epochs": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_metric": val_metrics,
    }


def load_predictions(pred_csv: str):
    with open(pred_csv) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        return [float(r[cols[-1]]) for r in reader]


def load_true_values(test_csv: str):
    with open(test_csv) as f:
        reader = csv.DictReader(f)
        return [float(r.get("Rf", r.get("y_mean", 0))) for r in reader]


def load_kermt_test_result(path: str):
    """KERMT fold_0/test_result.csv → (y_true, y_pred)，与训练时评估顺序一致。"""
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


def resolve_predictions_csv(pred_dir: str):
    """KERMT 预测可能在 model_0/ 或 fold_0/ 下。"""
    a = os.path.join(pred_dir, "predictions.csv")
    if os.path.isfile(a):
        return a
    b = os.path.join(os.path.dirname(pred_dir), "predictions.csv")
    if os.path.isfile(b):
        return b
    return None


def compute_metrics(y_true, y_pred):
    n = len(y_true)
    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)
    mean_t = sum(y_true) / n
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    ss_tot = sum((t - mean_t) ** 2 for t in y_true)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="*", default=[],
                        help="训练 log（如 nohup 重定向的 train.log）；可省略则只画散点")
    parser.add_argument("--labels", nargs="*", default=[],
                        help="与 --logs 一一对应；仅散点时可与 --pred_dirs 长度一致")
    parser.add_argument("--pred_dirs", nargs="+", default=None,
                        help="fold_0/model_0 等含 predictions.csv 的目录")
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument(
        "--test_result_csv",
        type=str,
        default=None,
        help="Plan A 训练生成的 fold_0/test_result.csv（推荐：与 predictions 行序一致，"
             "勿与全量 test.csv 按行 zip）",
    )
    parser.add_argument("--output", type=str, default="tlc/results/comparison.png")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. Install: pip install matplotlib")
        sys.exit(1)

    has_logs = bool(args.logs)
    has_test_result = bool(args.test_result_csv)
    has_preds = (args.pred_dirs is not None and args.test_csv is not None) or has_test_result

    if not has_logs and not has_preds:
        print("ERROR: 请提供 --logs/--labels 和/或 (--pred_dirs + --test_csv) 或 --test_result_csv",
              file=sys.stderr)
        sys.exit(1)
    if has_test_result and (args.pred_dirs is not None or args.test_csv is not None):
        print("ERROR: --test_result_csv 与 --pred_dirs/--test_csv 勿同时使用", file=sys.stderr)
        sys.exit(1)

    if has_logs and len(args.logs) != len(args.labels):
        print("ERROR: --logs 与 --labels 数量须一致", file=sys.stderr)
        sys.exit(1)

    if has_test_result and has_logs and len(args.logs) != 1:
        print("ERROR: 使用 --test_result_csv 时，--logs 最多 1 个（或省略 --logs 只画散点）",
              file=sys.stderr)
        sys.exit(1)

    if has_preds and not has_test_result:
        if not args.labels:
            args.labels = [f"run_{i}" for i in range(len(args.pred_dirs))]
        elif len(args.labels) != len(args.pred_dirs):
            print("ERROR: --labels 数量须与 --pred_dirs 一致（或省略 --labels）", file=sys.stderr)
            sys.exit(1)

    if has_test_result:
        if not args.labels:
            args.labels = ["test_result"]
        elif len(args.labels) != 1:
            print("ERROR: 使用 --test_result_csv 时 --labels 只接受一个名称", file=sys.stderr)
            sys.exit(1)

    if has_logs and has_preds and not has_test_result:
        if len(args.logs) != len(args.pred_dirs):
            print("ERROR: 同时画曲线与散点时，--logs 与 --pred_dirs 数量须一致（一一对应）",
                  file=sys.stderr)
            sys.exit(1)

    if has_preds:
        if has_logs:
            fig, axes = plt.subplots(2, 2, figsize=(14, 11))
            ax_tl, ax_vl, ax_vm = axes[0, 0], axes[0, 1], axes[1, 0]
            ax_scatter = axes[1, 1]
        else:
            fig, ax_scatter = plt.subplots(1, 1, figsize=(8, 8))
            ax_tl = ax_vl = ax_vm = None
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax_tl, ax_vl, ax_vm = axes[0], axes[1], axes[2]
        ax_scatter = None

    colors = plt.cm.tab10.colors

    if has_logs:
        for i, (log_path, label) in enumerate(zip(args.logs, args.labels)):
            data = parse_training_log(log_path)
            if not data["epochs"]:
                print(f"WARNING: No epochs found in {log_path}")
                continue
            c = colors[i % len(colors)]
            ax_tl.plot(data["epochs"], data["train_loss"], "-o", color=c, label=label, markersize=4)
            ax_vl.plot(data["epochs"], data["val_loss"], "-s", color=c, label=label, markersize=4)
            ax_vm.plot(data["epochs"], data["val_metric"], "-^", color=c, label=label, markersize=4)

        ax_tl.set_xlabel("Epoch")
        ax_tl.set_ylabel("Train Loss")
        ax_tl.set_title("Training Loss")
        ax_tl.legend()
        ax_tl.grid(True, alpha=0.3)

        ax_vl.set_xlabel("Epoch")
        ax_vl.set_ylabel("Val Loss")
        ax_vl.set_title("Validation Loss")
        ax_vl.legend()
        ax_vl.grid(True, alpha=0.3)

        ax_vm.set_xlabel("Epoch")
        ax_vm.set_ylabel("Val MAE")
        ax_vm.set_title("Validation MAE")
        ax_vm.legend()
        ax_vm.grid(True, alpha=0.3)

    if has_preds:
        if has_test_result:
            tr_path = args.test_result_csv
            if not os.path.isabs(tr_path):
                tr_path = os.path.normpath(_KERMT_ROOT / tr_path)
            y_true, y_pred = load_kermt_test_result(tr_path)
            n = len(y_true)
            metrics = compute_metrics(y_true, y_pred)
            label = args.labels[0]
            print(f"  [{label}] {tr_path}  n={n}  MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}")
            c = colors[0]
            ax_scatter.scatter(
                y_true, y_pred,
                alpha=0.3, s=10, color=c,
                label=f"{label}\nMAE={metrics['mae']:.4f} R²={metrics['r2']:.4f}"
            )
        else:
            test_csv = args.test_csv
            if not os.path.isabs(test_csv):
                test_csv = os.path.normpath(_KERMT_ROOT / test_csv)
            y_true = load_true_values(test_csv)
            for i, pred_dir in enumerate(args.pred_dirs):
                if not os.path.isabs(pred_dir):
                    pred_dir = os.path.normpath(_KERMT_ROOT / pred_dir)
                pred_csv = resolve_predictions_csv(pred_dir)
                label = args.labels[i]
                if not pred_csv:
                    print(f"WARNING: predictions.csv not found under {pred_dir} or its parent, skip")
                    continue
                y_pred = load_predictions(pred_csv)
                n = min(len(y_true), len(y_pred))
                if n < len(y_true) or n < len(y_pred):
                    print(
                        f"WARNING: 行数不一致 test_csv={len(y_true)} vs pred={len(y_pred)}，"
                        f"仅用前 {n} 行对齐；若指标异常请用 --test_result_csv 或重新 predict",
                    )
                metrics = compute_metrics(y_true[:n], y_pred[:n])
                print(f"  [{label}] {pred_csv}  n={n}  MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}")

                c = colors[i % len(colors)]
                ax_scatter.scatter(
                    y_true[:n], y_pred[:n],
                    alpha=0.3, s=10, color=c,
                    label=f"{label}\nMAE={metrics['mae']:.4f} R²={metrics['r2']:.4f}"
                )

        ax_scatter.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y=x")
        ax_scatter.set_xlabel("True Rf")
        ax_scatter.set_ylabel("Predicted Rf")
        ax_scatter.set_title("Test Set: True vs Predicted")
        ax_scatter.legend(fontsize=8)
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.set_xlim(-0.05, 1.05)
        ax_scatter.set_ylim(-0.05, 1.05)
        ax_scatter.set_aspect("equal")

    plt.tight_layout()
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()
