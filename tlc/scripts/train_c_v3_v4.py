#!/usr/bin/env python
"""
方案 C v3 / v4 / v4_d2 训练：
  v3   — 极简向量回归（单 MLP 头）；
  v4   — 双因果 MLP 解码 + teacher-forcing / 自回归推理（KermtDualCausalCv4）；
  v4_d2 — C v4 目标架构：共享 TransformerEncoder 因果双塔 + 逐步 ensemble（KermtCv4D2Transformer）。

用法:
  cd KERMT && python tlc/scripts/train_c_v3_v4.py --config tlc/configs/tlc_c_v3_vector_ep200_es100_bs128.yaml
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

from autoregressive_model import load_kermt_encoder
from c_v3_c_v4_model import KermtDualCausalCv4, KermtVectorCv3, regression_loss_tensor
from c_v4_d2_transformer import KermtCv4D2Transformer, align_steps_to_k


def regression_loss_masked(pred, target, mask, kind: str):
    """pred/target [K]；mask [K] 为 True 的位置参与损失。"""
    import torch.nn.functional as F

    if not mask.any():
        return pred.sum() * 0.0  # 保持计算图与 pred 同设备
    p = pred[mask]
    t = target[mask]
    if kind == "mse":
        return F.mse_loss(p, t)
    if kind == "mae":
        return F.l1_loss(p, t)
    raise ValueError(kind)


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


def train_epoch_v3(model, dataloader, optimizer, device, shared_dict, args, regression_loss: str):
    model.train()
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

        batch_loss = torch.tensor(0.0, device=device)
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

            pred = model(mol_vec, solvent_seq, desc_seq, solvent_vecs)
            loss = regression_loss_tensor(pred, rf_true, regression_loss)
            T = rf_true.numel()
            batch_loss = batch_loss + loss * T
            batch_steps += T

        if batch_steps > 0:
            avg_loss = batch_loss / batch_steps
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += avg_loss.item() * batch_steps
            total_steps += batch_steps

    return total_loss / max(total_steps, 1)


def train_epoch_v4(model, dataloader, optimizer, device, shared_dict, args, regression_loss: str, branch_aux: float):
    model.train()
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

        batch_loss = torch.tensor(0.0, device=device)
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

            rf_fwd, rf_bwd = model.forward_train(
                mol_vec, solvent_seq, desc_seq, rf_true, solvent_vecs
            )
            ens = 0.5 * (rf_fwd + rf_bwd)
            loss = regression_loss_tensor(ens, rf_true, regression_loss)
            if branch_aux > 0:
                loss = loss + branch_aux * (
                    regression_loss_tensor(rf_fwd, rf_true, regression_loss)
                    + regression_loss_tensor(rf_bwd, rf_true, regression_loss)
                )
            T = rf_true.numel()
            batch_loss = batch_loss + loss * T
            batch_steps += T

        if batch_steps > 0:
            avg_loss = batch_loss / batch_steps
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += avg_loss.item() * batch_steps
            total_steps += batch_steps

    return total_loss / max(total_steps, 1)


def train_epoch_v4_d2(model, dataloader, optimizer, device, shared_dict, args, regression_loss: str):
    """C v4 D2：逐步监督；num_channels=K>0 时为固定 K 通道多任务 Rf 向量（与 v3 目标一致），带掩码。"""
    model.train()
    total_loss = 0.0
    total_steps = 0
    K = int(getattr(model, "num_channels", 0) or 0)

    for batch_seqs in dataloader:
        unique_smiles = list({s["smiles"] for s in batch_seqs})
        smi_to_idx = {s: i for i, s in enumerate(unique_smiles)}

        graph_batch = build_graph_batch(unique_smiles, shared_dict, args)
        _, _, _, _, _, a_scope, _, _ = graph_batch
        a_scope_list = a_scope.data.cpu().numpy().tolist()

        mol_vecs = model.encode_molecules(graph_batch, a_scope_list)
        solvent_vecs = model.encode_solvents()

        batch_loss = torch.tensor(0.0, device=device)
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

            pred = model(mol_vec, solvent_seq, desc_seq, solvent_vecs)
            if K > 0:
                rf_k, _, _, valid = align_steps_to_k(rf_true, solvent_seq, desc_seq, K)
                loss = regression_loss_masked(pred, rf_k, valid, regression_loss)
                nv = int(valid.sum().item())
            else:
                loss = regression_loss_tensor(pred, rf_true, regression_loss)
                nv = rf_true.numel()
            batch_loss = batch_loss + loss * nv
            batch_steps += nv

        if batch_steps > 0:
            avg_loss = batch_loss / batch_steps
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += avg_loss.item() * batch_steps
            total_steps += batch_steps

    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate_v3(model, dataloader, device, shared_dict, args, regression_loss: str):
    model.eval()
    all_true = []
    all_pred = []
    total_obj = 0.0
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

            pred = model(mol_vec, solvent_seq, desc_seq, solvent_vecs)
            loss = regression_loss_tensor(pred, rf_true, regression_loss)
            T = rf_true.numel()
            total_obj += loss.item() * T
            total_steps += T

            all_true.extend(rf_true.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())

    return _metrics_from_lists(all_true, all_pred, total_obj, total_steps)


@torch.no_grad()
def evaluate_v4(model, dataloader, device, shared_dict, args, regression_loss: str):
    model.eval()
    all_true = []
    all_pred = []
    total_obj = 0.0
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

            rf_pred, _, _ = model.predict_sequence(mol_vec, solvent_seq, desc_seq, solvent_vecs)
            loss = regression_loss_tensor(rf_pred, rf_true, regression_loss)
            T = rf_true.numel()
            total_obj += loss.item() * T
            total_steps += T

            all_true.extend(rf_true.cpu().tolist())
            all_pred.extend(rf_pred.cpu().tolist())

    return _metrics_from_lists(all_true, all_pred, total_obj, total_steps)


@torch.no_grad()
def evaluate_v4_d2(model, dataloader, device, shared_dict, args, regression_loss: str):
    model.eval()
    all_true = []
    all_pred = []
    total_obj = 0.0
    total_steps = 0
    K = int(getattr(model, "num_channels", 0) or 0)

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

            pred = model(mol_vec, solvent_seq, desc_seq, solvent_vecs)
            if K > 0:
                rf_k, _, _, valid = align_steps_to_k(rf_true, solvent_seq, desc_seq, K)
                loss = regression_loss_masked(pred, rf_k, valid, regression_loss)
                nv = int(valid.sum().item())
                total_obj += loss.item() * nv
                total_steps += nv
                vm = valid.cpu().numpy().astype(bool)
                all_true.extend(rf_k.cpu().numpy()[vm].tolist())
                all_pred.extend(pred.cpu().numpy()[vm].tolist())
            else:
                loss = regression_loss_tensor(pred, rf_true, regression_loss)
                T = rf_true.numel()
                total_obj += loss.item() * T
                total_steps += T
                all_true.extend(rf_true.cpu().tolist())
                all_pred.extend(pred.cpu().tolist())

    return _metrics_from_lists(all_true, all_pred, total_obj, total_steps)


def _metrics_from_lists(all_true, all_pred, total_obj, total_steps):
    avg_obj = total_obj / max(total_steps, 1)
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
    "plan": "v3",
    "checkpoint": "pretrained_models/grover_base.pt",
    "train_seq": "tlc/data/train_sequences.json",
    "val_seq": "tlc/data/valid_sequences.json",
    "test_seq": "tlc/data/test_sequences.json",
    "save_dir": "tlc/results/c_v3_vector",
    "epochs": 200,
    "batch_size": 128,
    "lr": 1e-4,
    "encoder_lr": 1e-5,
    "weight_decay": 1e-5,
    "freeze_encoder": False,
    "seed": 42,
    "early_stop_epoch": 100,
    "n_solvents": 5,
    "desc_dim": 6,
    "mlp_hidden": 512,
    "mlp_blocks": 3,
    "dropout": 0.1,
    "regression_loss": "mae",
    "branch_aux_weight": 0.25,
    "transformer_d_model": 256,
    "transformer_nhead": 4,
    "transformer_nlayers": 2,
    "transformer_dim_ff": 0,
    "num_channels": 0,
    "bond_drop_rate": 0.0,
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
        elif key == "plan":
            parser.add_argument(
                f"--{key}",
                default=cfg.get(key, default),
                choices=("v3", "v4", "v4_d2"),
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
    if cli.regression_loss not in ("mse", "mae"):
        raise SystemExit("regression_loss must be mse or mae")
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
        bond_drop_rate=float(cli.bond_drop_rate),
        no_cache=True,
        use_cuikmolmaker_featurization=False,
    )

    print("Loading pretrained KERMT encoder...", flush=True)
    encoder, readout, mol_dim = load_kermt_encoder(cli.checkpoint, kermt_args)
    print(f"  mol_dim = {mol_dim}", flush=True)

    if cli.plan == "v3":
        model = KermtVectorCv3(
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
    elif cli.plan == "v4_d2":
        dim_ff = cli.transformer_dim_ff if cli.transformer_dim_ff > 0 else None
        model = KermtCv4D2Transformer(
            kermt_encoder=encoder,
            readout=readout,
            mol_dim=mol_dim,
            graph_args=kermt_args,
            desc_dim=cli.desc_dim,
            n_solvents=cli.n_solvents,
            d_model=cli.transformer_d_model,
            nhead=cli.transformer_nhead,
            num_layers=cli.transformer_nlayers,
            dim_feedforward=dim_ff,
            dropout=cli.dropout,
            num_channels=int(cli.num_channels),
        ).to(device)
    else:
        model = KermtDualCausalCv4(
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
    print(f"  Plan C {cli.plan.upper()} training", flush=True)
    if cli.plan == "v4_d2":
        print(
            "  Architecture: KermtCv4D2Transformer (D2 shared causal TransformerEncoder, "
            "fwd + rev causal towers, step ensemble mean)",
            flush=True,
        )
    print("=" * 60, flush=True)
    print(f"  Epochs: {cli.epochs}  batch: {cli.batch_size}", flush=True)
    print(f"  regression_loss={cli.regression_loss}", flush=True)
    if cli.plan == "v4":
        print(f"  branch_aux_weight={cli.branch_aux_weight}", flush=True)
    if cli.plan == "v4_d2":
        dff = cli.transformer_dim_ff if cli.transformer_dim_ff > 0 else f"4*{cli.transformer_d_model}"
        print(
            f"  transformer: d_model={cli.transformer_d_model} nhead={cli.transformer_nhead} "
            f"nlayers={cli.transformer_nlayers} dim_ff={dff}",
            flush=True,
        )
        kch = int(cli.num_channels)
        kmsg = f"fixed K={kch} multitask Rf vector" if kch > 0 else "variable len(steps)"
        print(f"  num_channels={kch} ({kmsg})", flush=True)
    print("=" * 60, flush=True)

    best_val_mae = float("inf")
    best_epoch = 0

    for epoch in range(cli.epochs):
        t0 = time.time()
        if cli.plan == "v3":
            train_loss = train_epoch_v3(
                model, train_loader, optimizer, device, shared_dict, kermt_args, cli.regression_loss
            )
            t_train = time.time() - t0
            t0 = time.time()
            val_metrics = evaluate_v3(
                model, val_loader, device, shared_dict, kermt_args, cli.regression_loss
            )
        elif cli.plan == "v4_d2":
            train_loss = train_epoch_v4_d2(
                model, train_loader, optimizer, device, shared_dict, kermt_args, cli.regression_loss
            )
            t_train = time.time() - t0
            t0 = time.time()
            val_metrics = evaluate_v4_d2(
                model, val_loader, device, shared_dict, kermt_args, cli.regression_loss
            )
        else:
            train_loss = train_epoch_v4(
                model,
                train_loader,
                optimizer,
                device,
                shared_dict,
                kermt_args,
                cli.regression_loss,
                cli.branch_aux_weight,
            )
            t_train = time.time() - t0
            t0 = time.time()
            val_metrics = evaluate_v4(
                model, val_loader, device, shared_dict, kermt_args, cli.regression_loss
            )
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
                    "plan": cli.plan,
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
