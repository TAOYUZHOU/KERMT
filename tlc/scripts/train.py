#!/usr/bin/env python
"""
KERMT TLC 训练入口 — 从 config.yaml 读取超参数

用法:
  python tlc/scripts/train.py --config tlc/configs/default.yaml
  python tlc/scripts/train.py --config tlc/configs/default.yaml --epochs 20 --batch_size 64
  python tlc/scripts/train.py --config tlc/configs/default.yaml --no_cuda

命令行参数会覆盖 yaml 中的同名字段。
"""

import argparse
import os
import sys
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from rdkit import RDLogger

KERMT_ROOT = str(Path(__file__).resolve().parents[2])
if KERMT_ROOT not in sys.path:
    sys.path.insert(0, KERMT_ROOT)

SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from config_loader import build_args, resolve_paths, load_yaml
from kermt.util.utils import create_logger
from kermt.data.torchvocab import MolVocab
from task.cross_validate import cross_validate


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True)


def main():
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    _ = MolVocab

    # nohup/重定向到文件时默认块缓冲，长时间看不到 print；改为行缓冲便于 tail 日志
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (OSError, ValueError):
            pass

    cli_parser = argparse.ArgumentParser(add_help=False)
    cli_parser.add_argument("--config", type=str, required=True,
                            help="Path to YAML config file")
    cli_known, cli_rest = cli_parser.parse_known_args()

    config_path = cli_known.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(KERMT_ROOT, config_path)

    args = build_args(
        config_path=config_path,
        cli_overrides=cli_rest,
        subcommand="finetune",
    )
    resolve_paths(args, KERMT_ROOT)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    shutil.copy2(config_path, os.path.join(save_dir, "config.yaml"))
    effective = {k: v for k, v in vars(args).items()
                 if not k.startswith("_") and v is not None}
    with open(os.path.join(save_dir, "effective_config.yaml"), "w") as f:
        yaml.dump(effective, f, default_flow_style=False, allow_unicode=True)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    setup_seed(args.seed)

    print("=" * 56, flush=True)
    print("  KERMT TLC Training (config-driven)", flush=True)
    print("=" * 56, flush=True)
    print(f"  Config:     {config_path}", flush=True)
    print(f"  Save dir:   {save_dir}", flush=True)
    print(f"  Epochs:     {args.epochs}", flush=True)
    print(f"  Batch size: {args.batch_size}", flush=True)
    print(f"  LR:         {args.init_lr} → {args.max_lr} → {args.final_lr}", flush=True)
    print(f"  FFN:        {args.ffn_num_layers} layers × {args.ffn_hidden_size}", flush=True)
    print(f"  Dropout:    {args.dropout}", flush=True)
    solv_d = getattr(args, 'solvent_emb_dim', 0)
    print(f"  Solvent emb:{f' {solv_d}d (learnable weighted)' if solv_d else ' off (raw features)'}", flush=True)
    if args.dataset_type == "regression":
        print(f"  Reg. loss:  {getattr(args, 'regression_loss', 'mse')}", flush=True)
    print(f"  Early stop: {args.early_stop_epoch} epochs without val improvement", flush=True)
    print(f"  Device:     {'CPU' if not args.cuda else f'GPU {args.gpu}'}", flush=True)
    print("=" * 56, flush=True)

    logger = create_logger(name="train", save_dir=save_dir, quiet=False)
    cross_validate(args, logger)

    print(flush=True)
    print(f"Training complete. Model saved to: {save_dir}", flush=True)


if __name__ == "__main__":
    main()
