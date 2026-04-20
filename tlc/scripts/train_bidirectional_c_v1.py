#!/usr/bin/env python
"""
方案 C v1 训练入口（见 tlc/README.md）。

用法:
  cd KERMT && python tlc/scripts/train_bidirectional_c_v1.py --config tlc/configs/tlc_bidirectional_c_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

KERMT_ROOT = str(Path(__file__).resolve().parents[2])
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
for _p in (KERMT_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from argparse import Namespace

import torch.nn.functional as F

from autoregressive_model import beta_nll_loss, load_kermt_encoder
from bidirectional_c_model import KermtBidirectionalCv1, scheme_c_loss


class SequenceDataset(Dataset):
    def __init__(self, seq_json_path):
        with open(seq_json_path) as f:
            self.sequences = json.load(f)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_sequences(batch):
    return batch


def build_graph_batch(smiles_list, shared_dict, args):
    from kermt.data.molgraph import mol2graph

    batch_mol_graph = mol2graph(smiles_list, shared_dict, args)
    return batch_mol_graph.get_components()


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    shared_dict,
    args,
    lambda_fwd,
    lambda_rev,
    regression_loss: str,
):
    model.train()
    total_loss = 0.0
    total_nll = 0.0
    total_steps = 0

    for batch_seqs in dataloader:
        unique_smiles = list({s["smiles"] for s in batch_seqs})
        smi_to_idx = {s: i for i, s in enumerate(unique_smiles)}

        graph_batch = build_graph_batch(unique_smiles, shared_dict, args)
        _, _, _, _, _, a_scope, _, _ = graph_batch
        a_scope_list = a_scope.data.cpu().numpy().tolist()

        mol_vecs = model.encode_molecules(graph_batch, a_scope_list)
        solvent_vecs = model.encode_solvents()

        batch_loss = torch.tensor(0.0, device=device)
        batch_nll = torch.tensor(0.0, device=device)
        batch_steps = 0

        for seq in batch_seqs:
            smi_idx = smi_to_idx[seq["smiles"]]
            mol_vec = mol_vecs[smi_idx]
            steps = seq["steps"]
            if len(steps) == 0:
                continue
            solvent_seq = torch.tensor(
                [s["solvent"] for s in steps], dtype=torch.float32, device=device
            )
            desc_seq = torch.tensor(
                [s["desc"] for s in steps], dtype=torch.float32, device=device
            )
            rf_true = torch.tensor(
                [s["rf"] for s in steps], dtype=torch.float32, device=device
            )

            mu, phi = model.forward_train(
                mol_vec, solvent_seq, desc_seq, rf_true, solvent_vecs
            )
            loss, main_term, _, _ = scheme_c_loss(
                mu, phi, rf_true, lambda_fwd, lambda_rev, regression_loss
            )
            T = rf_true.numel()
            batch_loss = batch_loss + loss * T
            batch_nll = batch_nll + main_term * T
            batch_steps += T

        if batch_steps > 0:
            avg_loss = batch_loss / batch_steps
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += avg_loss.item() * batch_steps
            total_nll += (batch_nll / batch_steps).item() * batch_steps  # main loss sum (name kept for log)
            total_steps += batch_steps

    return total_loss / max(total_steps, 1), total_nll / max(total_steps, 1)


@torch.no_grad()
def evaluate_epoch(
    model,
    dataloader,
    device,
    shared_dict,
    args,
    eval_refine: int,
    regression_loss: str,
):
    model.eval()
    all_true = []
    all_pred = []
    total_nll = 0.0
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
            if len(steps) == 0:
                continue
            solvent_seq = torch.tensor(
                [s["solvent"] for s in steps], dtype=torch.float32, device=device
            )
            desc_seq = torch.tensor(
                [s["desc"] for s in steps], dtype=torch.float32, device=device
            )
            rf_true = torch.tensor(
                [s["rf"] for s in steps], dtype=torch.float32, device=device
            )

            rf_pred, mu, phi = model.predict_sequence(
                mol_vec, solvent_seq, desc_seq, solvent_vecs, n_refine=eval_refine
            )
            if regression_loss == "beta_nll":
                loss = beta_nll_loss(mu, phi, rf_true)
            elif regression_loss == "mse":
                loss = F.mse_loss(rf_pred, rf_true)
            elif regression_loss == "mae":
                loss = F.l1_loss(rf_pred, rf_true)
            else:
                raise ValueError(regression_loss)
            T = rf_true.numel()
            total_nll += loss.item() * T
            total_steps += T

            all_true.extend(rf_true.cpu().tolist())
            all_pred.extend(rf_pred.cpu().tolist())

    avg_obj = total_nll / max(total_steps, 1)
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "loss": avg_obj,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "y_true": y_true,
        "y_pred": y_pred,
    }


_DEFAULTS = {
    "checkpoint": "pretrained_models/grover_base.pt",
    "train_seq": "tlc/data/train_sequences.json",
    "val_seq": "tlc/data/valid_sequences.json",
    "test_seq": "tlc/data/test_sequences.json",
    "save_dir": "tlc/results/bidirectional_c_v1",
    "epochs": 10,
    "batch_size": 32,
    "lr": 1e-4,
    "encoder_lr": 1e-5,
    "weight_decay": 1e-5,
    "freeze_encoder": False,
    "seed": 42,
    "early_stop_epoch": 20,
    "teacher_forcing": 1.0,
    "n_solvents": 5,
    "desc_dim": 6,
    "mlp_hidden": 512,
    "mlp_blocks": 3,
    "dropout": 0.1,
    "lambda_fwd": 0.1,
    "lambda_rev": 0.1,
    "eval_refine": 5,
    "regression_loss": "beta_nll",
}

_BOOL_FLAGS = {"freeze_encoder"}
_PATH_KEYS = {"checkpoint", "train_seq", "val_seq", "test_seq", "save_dir"}


def _parse_args():
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
            cfg.update(yaml.safe_load(f) or {})

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    for key, default in _DEFAULTS.items():
        if key in _BOOL_FLAGS:
            parser.add_argument(
                f"--{key}",
                default=cfg.get(key, default),
                type=lambda x: str(x).lower() in ("true", "1", "yes"),
                nargs="?",
                const=True,
            )
        else:
            parser.add_argument(
                f"--{key}",
                default=cfg.get(key, default),
                type=type(default) if default is not None else str,
            )
    args = parser.parse_args(rest)
    args._config_path = config_path

    for key in _PATH_KEYS:
        v = getattr(args, key, None)
        if v is not None and not os.path.isabs(v):
            setattr(args, key, os.path.join(KERMT_ROOT, v))

    return args


def main():
    cli = _parse_args()
    if getattr(cli, "regression_loss", "beta_nll") not in ("beta_nll", "mse", "mae"):
        raise SystemExit(
            f"regression_loss must be beta_nll, mse, or mae; got {cli.regression_loss!r}"
        )
    os.makedirs(cli.save_dir, exist_ok=True)

    if cli._config_path:
        shutil.copy2(cli._config_path, os.path.join(cli.save_dir, "config.yaml"))
    effective = {k: v for k, v in vars(cli).items() if not k.startswith("_")}
    with open(os.path.join(cli.save_dir, "effective_config.yaml"), "w") as f:
        yaml.dump(effective, f, default_flow_style=False, allow_unicode=True)

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    random.seed(cli.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

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

    print("Loading pretrained KERMT encoder...", flush=True)
    encoder, readout, mol_dim = load_kermt_encoder(cli.checkpoint, kermt_args)
    print(f"  mol_dim = {mol_dim}", flush=True)

    model = KermtBidirectionalCv1(
        kermt_encoder=encoder,
        readout=readout,
        mol_dim=mol_dim,
        graph_args=kermt_args,
        desc_dim=cli.desc_dim,
        n_solvents=cli.n_solvents,
        mlp_hidden=cli.mlp_hidden,
        mlp_blocks=cli.mlp_blocks,
        dropout=cli.dropout,
    ).to(device)

    if cli.freeze_encoder:
        for p in model.kermt_encoder.parameters():
            p.requires_grad = False
        for p in model.readout.parameters():
            p.requires_grad = False
        print("  Encoder frozen", flush=True)

    enc_params = list(model.kermt_encoder.parameters()) + list(model.readout.parameters())
    head_params = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("kermt_encoder.") and not n.startswith("readout.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": cli.encoder_lr},
            {"params": head_params, "lr": cli.lr},
        ],
        weight_decay=cli.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cli.epochs, 1), eta_min=1e-6
    )

    train_ds = SequenceDataset(cli.train_seq)
    val_ds = SequenceDataset(cli.val_seq)
    train_loader = DataLoader(
        train_ds,
        batch_size=cli.batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cli.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=0,
    )

    shared_dict = {}
    log_path = os.path.join(cli.save_dir, "train.log")
    log_f = open(log_path, "w")

    print("\n" + "=" * 60, flush=True)
    print("  Bidirectional C v1 Training (方案 C)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Epochs: {cli.epochs}  batch: {cli.batch_size}", flush=True)
    print(f"  regression_loss={cli.regression_loss}", flush=True)
    print(
        f"  lambda_fwd={cli.lambda_fwd} lambda_rev={cli.lambda_rev} eval_refine={cli.eval_refine}",
        flush=True,
    )
    print("=" * 60, flush=True)

    best_val_mae = float("inf")
    best_epoch = 0

    for epoch in range(cli.epochs):
        t0 = time.time()
        train_loss, train_main = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            shared_dict,
            kermt_args,
            cli.lambda_fwd,
            cli.lambda_rev,
            cli.regression_loss,
        )
        t_train = time.time() - t0

        t0 = time.time()
        val_metrics = evaluate_epoch(
            model,
            val_loader,
            device,
            shared_dict,
            kermt_args,
            cli.eval_refine,
            cli.regression_loss,
        )
        t_val = time.time() - t0

        scheduler.step()

        line = (
            f"Epoch: {epoch:04d} "
            f"loss_train: {train_loss:.6f} "
            f"main_train: {train_main:.6f} "
            f"loss_val: {val_metrics['loss']:.6f} "
            f"mae_val: {val_metrics['mae']:.4f} "
            f"rmse_val: {val_metrics['rmse']:.4f} "
            f"r2_val: {val_metrics['r2']:.4f} "
            f"cur_lr: {scheduler.get_last_lr()[0]:.6f} "
            f"t_time: {t_train:.1f}s v_time: {t_val:.1f}s"
        )
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": val_metrics["mae"],
                    "args": vars(cli),
                },
                os.path.join(cli.save_dir, "best_model.pt"),
            )

        if epoch - best_epoch >= cli.early_stop_epoch:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    print(f"\nBest val MAE: {best_val_mae:.4f} at epoch {best_epoch}", flush=True)
    log_f.write(f"\nBest val MAE: {best_val_mae:.4f} at epoch {best_epoch}\n")
    log_f.close()
    print(f"Training log: {log_path}", flush=True)
    print(f"Model saved to: {cli.save_dir}", flush=True)


if __name__ == "__main__":
    main()
