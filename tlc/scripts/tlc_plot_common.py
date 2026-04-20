"""
TLC 评测图共用逻辑：方案 A / 方案 B / 方案 C 共用 parity、combined 版式；
training_curves 按各方案 train.log 格式分函数解析。
"""
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"r2": r2, "mae": mae, "rmse": rmse, "n": len(y_true)}


def plot_parity_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str,
    split_name: str,
    timestamp: str,
    *,
    title_prefix: str,
    color: str = "#1f77b4",
    dpi: int = 200,
) -> Dict:
    """单 split parity：与 eval_and_plot_autoregressive 版式一致。"""
    m = metrics_arrays(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.35, s=12, edgecolors="none", c=color)

    lo = float(min(y_true.min(), y_pred.min())) - 0.02
    hi = float(max(y_true.max(), y_pred.max())) + 0.02
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, alpha=0.8, label="y = x")

    ax.set_xlabel("Experimental Rf", fontsize=13)
    ax.set_ylabel("Predicted Rf", fontsize=13)
    ax.set_title(
        f"{title_prefix} — {split_name}\n"
        f"R²={m['r2']:.4f}   MAE={m['mae']:.4f}   RMSE={m['rmse']:.4f}   n={m['n']}",
        fontsize=12,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)

    fname = os.path.join(save_dir, f"parity_{split_name}_{timestamp}.png")
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"  [{split_name:10s}]  R²={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  n={m['n']}")
    print(f"               Plot → {fname}")
    return m


def plot_combined_parity(
    all_metrics: Dict[str, Dict],
    output_dir: str,
    timestamp: str,
    *,
    suptitle: str = "TLC — Parity Plots",
) -> None:
    """三联 parity（Train / Valid / Test），版式与方案 B 一致。"""
    pred_files = {}
    for split_name in ["Train", "Valid", "Test"]:
        csv_path = os.path.join(output_dir, f"{split_name.lower()}_predictions.csv")
        if os.path.isfile(csv_path):
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            pred_files[split_name] = (data[:, 0], data[:, 1])

    if len(pred_files) < 2:
        return

    n_plots = len(pred_files)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6.5))
    if n_plots == 1:
        axes = [axes]

    colors = {"Train": "#2196F3", "Valid": "#FF9800", "Test": "#4CAF50"}

    for ax, (split_name, (y_true, y_pred)) in zip(axes, pred_files.items()):
        m = all_metrics[split_name]
        c = colors.get(split_name, "#1f77b4")

        ax.scatter(y_true, y_pred, alpha=0.3, s=10, edgecolors="none", c=c)

        lo = min(y_true.min(), y_pred.min()) - 0.02
        hi = max(y_true.max(), y_pred.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Experimental Rf", fontsize=12)
        ax.set_ylabel("Predicted Rf", fontsize=12)
        ax.set_title(
            f"{split_name} (n={m['n']})\n"
            f"R²={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"parity_combined_{timestamp}.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Combined parity plot → {fname}")


def plot_training_curves_beta(log_path: str, save_path: str, timestamp: str) -> None:
    """方案 B train.log：Beta NLL + mae_val / rmse_val / r2_val。"""
    import re
    pattern = re.compile(
        r"Epoch:\s*(\d+)\s+"
        r"loss_train:\s*([\-\d.]+)\s+"
        r"loss_val:\s*([\-\d.]+)\s+"
        r"mae_val:\s*([\d.]+)\s+"
        r"rmse_val:\s*([\d.]+)\s+"
        r"r2_val:\s*([\d.]+)"
    )

    epochs, train_loss, val_loss = [], [], []
    val_mae, val_rmse, val_r2 = [], [], []

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_loss.append(float(m.group(3)))
                val_mae.append(float(m.group(4)))
                val_rmse.append(float(m.group(5)))
                val_r2.append(float(m.group(6)))

    if not epochs:
        print("  Warning: no training epochs parsed from log (Beta format)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_loss, label="Train Loss", alpha=0.8)
    axes[0].plot(epochs, val_loss, "--", label="Val Loss", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Beta NLL Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_mae, label="Val MAE", color="tab:orange", alpha=0.8)
    axes[1].plot(epochs, val_rmse, "--", label="Val RMSE", color="tab:red", alpha=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Validation MAE & RMSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, val_r2, label="Val R²", color="tab:green", alpha=0.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("R²")
    axes[2].set_title("Validation R²")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_path, f"training_curves_{timestamp}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves → {fname}")


def plot_training_curves_scheme_c(log_path: str, save_path: str, timestamp: str) -> None:
    """方案 C train.log：支持 main_train/loss_val（新）或 nll_train_approx/loss_val_nll（旧）。"""
    import re

    pattern_new = re.compile(
        r"Epoch:\s*(\d+)\s+"
        r"loss_train:\s*([-\d.eE+]+)\s+"
        r"main_train:\s*([-\d.eE+]+)\s+"
        r"loss_val:\s*([-\d.eE+]+)\s+"
        r"mae_val:\s*([-\d.eE+]+)\s+"
        r"rmse_val:\s*([-\d.eE+]+)\s+"
        r"r2_val:\s*([-\d.eE+]+)"
    )
    pattern_old = re.compile(
        r"Epoch:\s*(\d+)\s+"
        r"loss_train:\s*([-\d.eE+]+)\s+"
        r"nll_train_approx:\s*([-\d.eE+]+)\s+"
        r"loss_val_nll:\s*([-\d.eE+]+)\s+"
        r"mae_val:\s*([-\d.eE+]+)\s+"
        r"rmse_val:\s*([-\d.eE+]+)\s+"
        r"r2_val:\s*([-\d.eE+]+)"
    )

    epochs = []
    loss_train, main_train, loss_val = [], [], []
    val_mae, val_rmse, val_r2 = [], [], []

    with open(log_path) as f:
        for line in f:
            m = pattern_new.search(line) or pattern_old.search(line)
            if m:
                epochs.append(int(m.group(1)))
                loss_train.append(float(m.group(2)))
                main_train.append(float(m.group(3)))
                loss_val.append(float(m.group(4)))
                val_mae.append(float(m.group(5)))
                val_rmse.append(float(m.group(6)))
                val_r2.append(float(m.group(7)))

    if not epochs:
        print("  Warning: no training epochs parsed from log (scheme C format)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, loss_train, label="loss_train (total)", alpha=0.85)
    axes[0].plot(epochs, main_train, "--", label="main_train", alpha=0.85)
    axes[0].plot(epochs, loss_val, "-.", label="loss_val", alpha=0.85)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Scheme C — Train / Val objective")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_mae, label="Val MAE", color="tab:orange", alpha=0.8)
    axes[1].plot(epochs, val_rmse, "--", label="Val RMSE", color="tab:red", alpha=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Validation MAE & RMSE (stepwise Rf)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, val_r2, label="Val R²", color="tab:green", alpha=0.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("R²")
    axes[2].set_title("Validation R²")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_path, f"training_curves_{timestamp}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves → {fname}")


def plot_training_curves_c_v3_v4(log_path: str, save_path: str, timestamp: str) -> None:
    """train_c_v3_v4.py 的 train.log：无 main_train 字段。"""
    import re

    pattern = re.compile(
        r"Epoch:\s*(\d+)\s+"
        r"loss_train:\s*([-\d.eE+]+)\s+"
        r"loss_val:\s*([-\d.eE+]+)\s+"
        r"mae_val:\s*([-\d.eE+]+)\s+"
        r"rmse_val:\s*([-\d.eE+]+)\s+"
        r"r2_val:\s*([-\d.eE+]+)"
    )

    epochs = []
    loss_train, loss_val = [], []
    val_mae, val_rmse, val_r2 = [], [], []

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                loss_train.append(float(m.group(2)))
                loss_val.append(float(m.group(3)))
                val_mae.append(float(m.group(4)))
                val_rmse.append(float(m.group(5)))
                val_r2.append(float(m.group(6)))

    if not epochs:
        print("  Warning: no training epochs parsed from log (C v3/v4 format)")
        return

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, loss_train, label="loss_train", alpha=0.85)
    axes[0].plot(epochs, loss_val, "--", label="loss_val", alpha=0.85)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("C v3/v4 — Train / Val loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_mae, label="Val MAE", color="tab:orange", alpha=0.8)
    axes[1].plot(epochs, val_rmse, "--", label="Val RMSE", color="tab:red", alpha=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Validation MAE & RMSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, val_r2, label="Val R²", color="tab:green", alpha=0.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("R²")
    axes[2].set_title("Validation R²")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_path, f"training_curves_{timestamp}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves → {fname}")


def plot_training_curves_mse(log_path: str, save_path: str, timestamp: str) -> None:
    """方案 A cross_validate 日志：loss_train / loss_val / mae_val（无 rmse/r2 每行）。

    loss 允许负号（例如 Plan A + regression_loss=beta_nll 时 train 为 Beta NLL）。
    """
    import re
    pattern = re.compile(
        r"Epoch:\s*(\d+)\s+"
        r"loss_train:\s*([\-\d.]+)\s+"
        r"loss_val:\s*([\-\d.]+)\s+"
        r"mae_val:\s*([\d.]+)"
    )

    epochs, train_loss, val_loss, val_mae = [], [], [], []

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_loss.append(float(m.group(3)))
                val_mae.append(float(m.group(4)))

    if not epochs:
        print("  Warning: no training epochs parsed from log (Plan A epoch line format)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    neg_train = any(t < 0 for t in train_loss)
    loss_ylabel = "Loss (incl. Beta NLL)" if neg_train else "MSE Loss"

    axes[0].plot(epochs, train_loss, label="Train Loss", alpha=0.8)
    axes[0].plot(epochs, val_loss, "--", label="Val Loss", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(loss_ylabel)
    axes[0].set_title("Training & Validation Loss (Plan A)")
    # Train (e.g. Beta NLL) can be negative while val loss is MSE on μ and positive — autoscale would clip one branch.
    lo = min(min(train_loss), min(val_loss))
    hi = max(max(train_loss), max(val_loss))
    pad = 0.05 * (hi - lo) if hi > lo else 0.1
    axes[0].set_ylim(lo - pad, hi + pad)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_mae, label="Val MAE", color="tab:orange", alpha=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Validation MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(save_path, f"training_curves_{timestamp}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves → {fname}")
