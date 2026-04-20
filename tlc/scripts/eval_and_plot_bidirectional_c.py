#!/usr/bin/env python
"""
方案 C 评测绘图：与 `eval_and_plot_autoregressive.py --plan bidirectional_c` 等价；
亦可评测 **C v3 / C v4**（`train_c_v3_v4.py` 保存的 `best_model.pt`）。

用法:
  cd KERMT && python tlc/scripts/eval_and_plot_bidirectional_c.py --model_dir tlc/results/bidirectional_c_v1_ep200_bs128

  # C v3 / v4（需指定 plan）
  python tlc/scripts/eval_and_plot_bidirectional_c.py --plan c_v3 --model_dir tlc/results/c_v3_vector_datav2_ep200_es100_bs128
  python tlc/scripts/eval_and_plot_bidirectional_c.py --plan c_v4 --model_dir tlc/results/c_v4_dual_causal_datav2_ep200_es100_bs128
  python tlc/scripts/eval_and_plot_bidirectional_c.py --plan c_v4_d2 --model_dir tlc/results/c_v4_d2_datav2_ep200_es50_bs128
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

KERMT_ROOT = str(Path(__file__).resolve().parents[2])
SCRIPTS_DIR = str(Path(__file__).resolve().parent)
for _p in (KERMT_ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_and_plot_autoregressive import run_eval_and_plot


def main():
    p = argparse.ArgumentParser(
        description="Plan C bidirectional / C v3 / C v4 — parity plots (wrapper)"
    )
    p.add_argument(
        "--plan",
        type=str,
        choices=("bidirectional_c", "c_v3", "c_v4", "c_v4_d2"),
        default="bidirectional_c",
        help="bidirectional_c=KermtBidirectionalCv1；c_v3/c_v4/c_v4_d2=train_c_v3_v4 系列",
    )
    p.add_argument("--model_dir", type=str, default="tlc/results/bidirectional_c_v1")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--title_prefix", type=str, default=None)
    p.add_argument("--suptitle", type=str, default=None)
    args = p.parse_args()

    model_dir = args.model_dir
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(KERMT_ROOT, model_dir)

    run_eval_and_plot(
        plan=args.plan,
        model_dir=model_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gpu=args.gpu,
        title_prefix=args.title_prefix,
        suptitle=args.suptitle,
    )


if __name__ == "__main__":
    main()
