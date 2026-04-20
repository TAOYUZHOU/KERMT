#!/usr/bin/env python
"""
方案 B: 自回归单调 Rf 预测训练脚本

用法:
  # config-driven (推荐)
  python tlc/scripts/train_autoregressive.py --config tlc/configs/tlc_autoregressive.yaml
  python tlc/scripts/train_autoregressive.py --config tlc/configs/tlc_autoregressive.yaml --epochs 50

  # 纯 CLI (向后兼容)
  CUDA_VISIBLE_DEVICES=0 python tlc/scripts/train_autoregressive.py \
      --checkpoint pretrained_models/grover_base.pt \
      --train_seq tlc/data/train_sequences.json \
      --val_seq tlc/data/valid_sequences.json \
      --save_dir tlc/results/autoregressive \
      --epochs 10 --batch_size 16 --lr 1e-4
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

KERMT_ROOT = str(Path(__file__).resolve().parents[2])
if KERMT_ROOT not in sys.path:
    sys.path.insert(0, KERMT_ROOT)

from kermt.data.molgraph import mol2graph, MolGraph
from kermt.model.layers import Readout
from kermt.model.models import KERMTEmbedding
from autoregressive_model import (
    KermtAutoregressive, beta_nll_loss, load_kermt_encoder,
)


class SequenceDataset(Dataset):
    """Dataset that yields molecule sequences for autoregressive training."""

    def __init__(self, seq_json_path):
        with open(seq_json_path) as f:
            self.sequences = json.load(f)
        self.smiles_set = list({s["smiles"] for s in self.sequences})

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_sequences(batch):
    """
    Custom collator: extract unique SMILES and prepare sequence data.
    Returns raw dicts; graph batching happens in training step.
    """
    return batch


def build_graph_batch(smiles_list, shared_dict, args):
    """Build KERMT graph batch from a list of SMILES strings."""
    batch_mol_graph = mol2graph(smiles_list, shared_dict, args)
    return batch_mol_graph.get_components()


def train_epoch(model, dataloader, optimizer, device, shared_dict, args,
                teacher_forcing_ratio=1.0):
    model.train()
    total_loss = 0.0
    total_steps = 0
    n_batches = 0

    for batch_seqs in dataloader:
        unique_smiles = list({s["smiles"] for s in batch_seqs})
        smi_to_idx = {s: i for i, s in enumerate(unique_smiles)}

        graph_batch = build_graph_batch(unique_smiles, shared_dict, args)
        _, _, _, _, _, a_scope, _, _ = graph_batch
        a_scope_list = a_scope.data.cpu().numpy().tolist()

        mol_vecs = model.encode_molecules(graph_batch, a_scope_list)
        solvent_vecs = model.encode_solvents()

        batch_loss = torch.tensor(0.0, device=device)
        batch_step_count = 0

        for seq in batch_seqs:
            smi_idx = smi_to_idx[seq["smiles"]]
            mol_vec = mol_vecs[smi_idx]

            steps = seq["steps"]
            T = len(steps)
            if T == 0:
                continue

            solvent_seq = torch.tensor(
                [s["solvent"] for s in steps],
                dtype=torch.float32, device=device,
            )
            desc_seq = torch.tensor(
                [s["desc"] for s in steps],
                dtype=torch.float32, device=device,
            )
            rf_true = torch.tensor(
                [s["rf"] for s in steps],
                dtype=torch.float32, device=device,
            )
            boundaries = seq.get("system_boundaries", [0])

            rf_preds, mus, phis = model.predict_sequence(
                mol_vec, solvent_seq, desc_seq, boundaries,
                solvent_vecs=solvent_vecs,
                true_rf_seq=rf_true,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )

            loss = beta_nll_loss(mus, phis, rf_true)
            batch_loss = batch_loss + loss * T
            batch_step_count += T

        if batch_step_count > 0:
            avg_loss = batch_loss / batch_step_count
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += avg_loss.item() * batch_step_count
            total_steps += batch_step_count

        n_batches += 1

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, shared_dict, args):
    model.eval()
    all_true = []
    all_pred = []
    total_loss = 0.0
    total_steps = 0

    for batch_seqs in dataloader:
        unique_smiles = list({s["smiles"] for s in batch_seqs})
        smi_to_idx = {s: i for i, s in enumerate(unique_smiles)}

        graph_batch = build_graph_batch(unique_smiles, shared_dict, args)
        _, _, _, _, _, a_scope, _, _ = graph_batch
        a_scope_list = a_scope.data.cpu().numpy().tolist()

        mol_vecs = model.encode_molecules(graph_batch, a_scope_list)
        solvent_vecs = model.encode_solvents()

        for seq in batch_seqs:
            smi_idx = smi_to_idx[seq["smiles"]]
            mol_vec = mol_vecs[smi_idx]

            steps = seq["steps"]
            T = len(steps)
            if T == 0:
                continue

            solvent_seq = torch.tensor(
                [s["solvent"] for s in steps],
                dtype=torch.float32, device=device,
            )
            desc_seq = torch.tensor(
                [s["desc"] for s in steps],
                dtype=torch.float32, device=device,
            )
            rf_true = torch.tensor(
                [s["rf"] for s in steps],
                dtype=torch.float32, device=device,
            )
            boundaries = seq.get("system_boundaries", [0])

            rf_preds, mus, phis = model.predict_sequence(
                mol_vec, solvent_seq, desc_seq, boundaries,
                solvent_vecs=solvent_vecs,
                true_rf_seq=None,
                teacher_forcing_ratio=0.0,
            )

            loss = beta_nll_loss(mus, phis, rf_true)
            total_loss += loss.item() * T
            total_steps += T

            all_true.extend(rf_true.cpu().tolist())
            all_pred.extend(rf_preds.cpu().tolist())

    avg_loss = total_loss / max(total_steps, 1)

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "loss": avg_loss,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "y_true": y_true,
        "y_pred": y_pred,
    }


_DEFAULTS = {
    "checkpoint":       "pretrained_models/grover_base.pt",
    "train_seq":        "tlc/data/train_sequences.json",
    "val_seq":          "tlc/data/valid_sequences.json",
    "test_seq":         "tlc/data/test_sequences.json",
    "save_dir":         "tlc/results/autoregressive",
    "epochs":           10,
    "batch_size":       16,
    "lr":               1e-4,
    "encoder_lr":       1e-5,
    "weight_decay":     1e-5,
    "freeze_encoder":   False,
    "teacher_forcing":  1.0,
    "seed":             42,
    "early_stop_epoch": 10,
    "desc_dim":         6,
    "desc_emb_dim":     64,
    "n_solvents":       5,
    "ffn_hidden":       512,
    "ffn_blocks":       3,
    "dropout":          0.1,
}

_BOOL_FLAGS = {"freeze_encoder"}
_PATH_KEYS = {"checkpoint", "train_seq", "val_seq", "test_seq", "save_dir"}


def _parse_args():
    """Parse args: YAML config (optional) + CLI overrides."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, rest = pre.parse_known_args()

    cfg = dict(_DEFAULTS)
    config_path = None
    if known.config:
        config_path = known.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(KERMT_ROOT, config_path)
        with open(config_path) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        cfg.update(yaml_cfg)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    for key, default in _DEFAULTS.items():
        if key in _BOOL_FLAGS:
            parser.add_argument(f"--{key}", default=cfg.get(key, default),
                                type=lambda x: str(x).lower() in ("true", "1", "yes"),
                                nargs="?", const=True)
        else:
            parser.add_argument(f"--{key}", default=cfg.get(key, default),
                                type=type(default) if default is not None else str)
    args = parser.parse_args(rest)
    args._config_path = config_path

    for key in _PATH_KEYS:
        v = getattr(args, key, None)
        if v is not None and not os.path.isabs(v):
            setattr(args, key, os.path.join(KERMT_ROOT, v))

    return args


def main():
    cli_args = _parse_args()
    os.makedirs(cli_args.save_dir, exist_ok=True)

    if cli_args._config_path:
        shutil.copy2(cli_args._config_path, os.path.join(cli_args.save_dir, "config.yaml"))
    effective = {k: v for k, v in vars(cli_args).items() if not k.startswith("_")}
    with open(os.path.join(cli_args.save_dir, "effective_config.yaml"), "w") as f:
        yaml.dump(effective, f, default_flow_style=False, allow_unicode=True)

    torch.manual_seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    random.seed(cli_args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from argparse import Namespace
    kermt_args = Namespace(
        cuda=(device.type == "cuda"),
        dropout=0.0,
        activation="PReLU",
        self_attention=False,
        attn_hidden=4,
        attn_out=128,
        bond_drop_rate=0.0,
        no_cache=True,
        use_cuikmolmaker_featurization=False,
    )

    print("Loading pretrained KERMT encoder...")
    encoder, readout, mol_dim = load_kermt_encoder(cli_args.checkpoint, kermt_args)
    print(f"  mol_dim = {mol_dim}")

    model = KermtAutoregressive(
        kermt_encoder=encoder,
        readout=readout,
        mol_dim=mol_dim,
        graph_args=kermt_args,
        desc_dim=cli_args.desc_dim,
        desc_emb_dim=cli_args.desc_emb_dim,
        n_solvents=cli_args.n_solvents,
        ffn_hidden=cli_args.ffn_hidden,
        ffn_blocks=cli_args.ffn_blocks,
        dropout=cli_args.dropout,
    ).to(device)

    if cli_args.freeze_encoder:
        for p in model.kermt_encoder.parameters():
            p.requires_grad = False
        for p in model.readout.parameters():
            p.requires_grad = False
        print("  Encoder frozen")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {n_params:,}")
    print(f"  Trainable:    {n_trainable:,}")

    encoder_params = list(model.kermt_encoder.parameters()) + list(model.readout.parameters())
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("kermt_encoder.") and not n.startswith("readout.")]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cli_args.encoder_lr},
        {"params": head_params, "lr": cli_args.lr},
    ], weight_decay=cli_args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cli_args.epochs, eta_min=1e-6
    )

    print("\nLoading data...")
    train_ds = SequenceDataset(cli_args.train_seq)
    val_ds = SequenceDataset(cli_args.val_seq)
    print(f"  Train: {len(train_ds)} sequences")
    print(f"  Val:   {len(val_ds)} sequences")

    train_loader = DataLoader(
        train_ds, batch_size=cli_args.batch_size,
        shuffle=True, collate_fn=collate_sequences,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cli_args.batch_size,
        shuffle=False, collate_fn=collate_sequences,
        num_workers=0,
    )

    shared_dict = {}

    log_path = os.path.join(cli_args.save_dir, "train.log")
    log_f = open(log_path, "w")

    print("\n" + "=" * 60)
    print("  Autoregressive Monotonic Rf Training (方案 B)")
    print("=" * 60)
    print(f"  Epochs:       {cli_args.epochs}")
    print(f"  Batch size:   {cli_args.batch_size}")
    print(f"  Head LR:      {cli_args.lr}")
    print(f"  Encoder LR:   {cli_args.encoder_lr}")
    print(f"  TF ratio:     {cli_args.teacher_forcing}")
    print(f"  Solvent enc:  GNN ({cli_args.n_solvents} solvents → mol_dim → SolventProjection)")
    print(f"  Desc emb:     {cli_args.desc_emb_dim}")
    print(f"  FFN:          {cli_args.ffn_blocks} blocks × {cli_args.ffn_hidden}")
    print(f"  Early stop:   {cli_args.early_stop_epoch} epochs patience")
    print("=" * 60)

    best_val_mae = float("inf")
    best_epoch = 0

    for epoch in range(cli_args.epochs):
        t0 = time.time()

        tf_ratio = cli_args.teacher_forcing
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            shared_dict, kermt_args, tf_ratio,
        )
        t_train = time.time() - t0

        t0 = time.time()
        val_metrics = evaluate_epoch(model, val_loader, device, shared_dict, kermt_args)
        t_val = time.time() - t0

        scheduler.step()

        line = (
            f"Epoch: {epoch:04d} "
            f"loss_train: {train_loss:.6f} "
            f"loss_val: {val_metrics['loss']:.6f} "
            f"mae_val: {val_metrics['mae']:.4f} "
            f"rmse_val: {val_metrics['rmse']:.4f} "
            f"r2_val: {val_metrics['r2']:.4f} "
            f"cur_lr: {scheduler.get_last_lr()[0]:.6f} "
            f"t_time: {t_train:.1f}s "
            f"v_time: {t_val:.1f}s"
        )
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mae": val_metrics["mae"],
                "args": vars(cli_args),
            }, os.path.join(cli_args.save_dir, "best_model.pt"))

        if epoch - best_epoch >= cli_args.early_stop_epoch:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {cli_args.early_stop_epoch} epochs)")
            break

    print(f"\nBest val MAE: {best_val_mae:.4f} at epoch {best_epoch}")

    if cli_args.test_seq and os.path.isfile(cli_args.test_seq):
        print("\nEvaluating on test set...")
        ckpt = torch.load(os.path.join(cli_args.save_dir, "best_model.pt"),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        test_ds = SequenceDataset(cli_args.test_seq)
        test_loader = DataLoader(
            test_ds, batch_size=cli_args.batch_size,
            shuffle=False, collate_fn=collate_sequences,
            num_workers=0,
        )
        test_metrics = evaluate_epoch(model, test_loader, device, shared_dict, kermt_args)
        print(f"  Test MAE:  {test_metrics['mae']:.4f}")
        print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"  Test R²:   {test_metrics['r2']:.4f}")

        import csv
        pred_csv = os.path.join(cli_args.save_dir, "test_predictions.csv")
        with open(pred_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true_rf", "pred_rf"])
            for t, p in zip(test_metrics["y_true"], test_metrics["y_pred"]):
                writer.writerow([f"{t:.6f}", f"{p:.6f}"])
        print(f"  Predictions: {pred_csv}")

    log_f.close()
    print(f"\nTraining log: {log_path}")
    print(f"Model saved to: {cli_args.save_dir}")


if __name__ == "__main__":
    main()
