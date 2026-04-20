#!/usr/bin/env python
"""
序列 TLC 模型：加载 best_model.pt，在 Train/Valid/Test 上推理并生成与
`tlc/results/autoregressive_v3_solv_gnn_ep300_es100_bs128/plots` 相同布局的产物。

支持实现（--plan）：
  - autoregressive  方案 B：KermtAutoregressive / Legacy
  - bidirectional_c 方案 C：KermtBidirectionalCv1
  - c_v3 / c_v4      方案 C v3/v4：`train_c_v3_v4.py` 与 `c_v3_c_v4_model.py`

产物（均在 output_dir，默认 <model_dir>/plots）：
  train_predictions.csv, valid_predictions.csv, test_predictions.csv
  parity_Train|Valid|Test_<timestamp>.png
  parity_combined_<timestamp>.png
  training_curves_<timestamp>.png（若存在 train.log）

用法:
  cd KERMT

  # 方案 B（默认）
  python tlc/scripts/eval_and_plot_autoregressive.py \\
      --plan autoregressive \\
      --model_dir tlc/results/autoregressive_v3_solv_gnn_ep300_es100_bs128

  # 方案 C
  python tlc/scripts/eval_and_plot_autoregressive.py \\
      --plan bidirectional_c \\
      --model_dir tlc/results/bidirectional_c_v1_ep200_bs128

  CUDA_VISIBLE_DEVICES=0 python tlc/scripts/eval_and_plot_autoregressive.py --plan ... --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

KERMT_ROOT = str(Path(__file__).resolve().parents[2])
if KERMT_ROOT not in sys.path:
    sys.path.insert(0, KERMT_ROOT)

SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from argparse import Namespace

from kermt.data.molgraph import mol2graph
from autoregressive_model import (
    KermtAutoregressive,
    KermtAutoregressiveLegacy,
    load_kermt_encoder,
)
from tlc_plot_common import (
    plot_combined_parity,
    plot_parity_split,
    plot_training_curves_beta,
    plot_training_curves_c_v3_v4,
    plot_training_curves_scheme_c,
)


class SequenceDataset(Dataset):
    def __init__(self, seq_json_path):
        with open(seq_json_path) as f:
            self.sequences = json.load(f)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def _collate(batch):
    return batch


def _build_graph_batch(smiles_list, shared_dict, graph_args):
    batch_mol_graph = mol2graph(smiles_list, shared_dict, graph_args)
    return batch_mol_graph.get_components()


@torch.no_grad()
def predict_split_autoregressive(model, dataloader, device, shared_dict, kermt_args):
    model.eval()
    all_true, all_pred = [], []
    use_legacy = isinstance(model, KermtAutoregressiveLegacy)

    for batch_seqs in dataloader:
        unique_smiles = list({s["smiles"] for s in batch_seqs})
        smi_to_idx = {s: i for i, s in enumerate(unique_smiles)}

        batch_mol_graph = mol2graph(unique_smiles, shared_dict, kermt_args)
        graph_batch = batch_mol_graph.get_components()
        _, _, _, _, _, a_scope, _, _ = graph_batch
        a_scope_list = a_scope.data.cpu().numpy().tolist()

        mol_vecs = model.encode_molecules(graph_batch, a_scope_list)
        if not use_legacy:
            solvent_vecs = model.encode_solvents()

        for seq in batch_seqs:
            smi_idx = smi_to_idx[seq["smiles"]]
            mol_vec = mol_vecs[smi_idx]

            steps = seq["steps"]
            if len(steps) == 0:
                continue

            solvent_seq = torch.tensor(
                [s["solvent"] for s in steps],
                dtype=torch.float32,
                device=device,
            )
            desc_seq = torch.tensor(
                [s["desc"] for s in steps],
                dtype=torch.float32,
                device=device,
            )
            rf_true = torch.tensor(
                [s["rf"] for s in steps],
                dtype=torch.float32,
                device=device,
            )
            boundaries = seq.get("system_boundaries", [0])

            if use_legacy:
                cross_seq = torch.tensor(
                    [s.get("cross", [0.0] * 10) for s in steps],
                    dtype=torch.float32,
                    device=device,
                )
                rf_preds, _, _ = model.predict_sequence(
                    mol_vec,
                    solvent_seq,
                    cross_seq,
                    desc_seq,
                    boundaries,
                    true_rf_seq=None,
                    teacher_forcing_ratio=0.0,
                )
            else:
                rf_preds, _, _ = model.predict_sequence(
                    mol_vec,
                    solvent_seq,
                    desc_seq,
                    boundaries,
                    solvent_vecs=solvent_vecs,
                    true_rf_seq=None,
                    teacher_forcing_ratio=0.0,
                )

            all_true.extend(rf_true.cpu().tolist())
            all_pred.extend(rf_preds.cpu().tolist())

    return np.array(all_true), np.array(all_pred)


@torch.no_grad()
def predict_split_bidirectional(model, dataloader, device, shared_dict, kermt_args, eval_refine: int):
    from bidirectional_c_model import KermtBidirectionalCv1

    model.eval()
    all_true, all_pred = [], []
    assert isinstance(model, KermtBidirectionalCv1)

    for batch_seqs in dataloader:
        unique_smiles = list({s["smiles"] for s in batch_seqs})
        smi_to_idx = {s: i for i, s in enumerate(unique_smiles)}

        graph_batch = _build_graph_batch(unique_smiles, shared_dict, kermt_args)
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

            rf_pred, _, _ = model.predict_sequence(
                mol_vec, solvent_seq, desc_seq, solvent_vecs, n_refine=eval_refine
            )
            all_true.extend(rf_true.cpu().tolist())
            all_pred.extend(rf_pred.cpu().tolist())

    return np.array(all_true, dtype=np.float64), np.array(all_pred, dtype=np.float64)


@torch.no_grad()
def predict_split_c_v3(model, dataloader, device, shared_dict, kermt_args):
    from c_v3_c_v4_model import KermtVectorCv3

    model.eval()
    all_true, all_pred = [], []
    assert isinstance(model, KermtVectorCv3)

    for batch_seqs in dataloader:
        unique_smiles = list({s["smiles"] for s in batch_seqs})
        smi_to_idx = {s: i for i, s in enumerate(unique_smiles)}

        graph_batch = _build_graph_batch(unique_smiles, shared_dict, kermt_args)
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
            all_true.extend(rf_true.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())

    return np.array(all_true, dtype=np.float64), np.array(all_pred, dtype=np.float64)


@torch.no_grad()
def predict_split_c_v4(model, dataloader, device, shared_dict, kermt_args):
    from c_v3_c_v4_model import KermtDualCausalCv4

    model.eval()
    all_true, all_pred = [], []
    assert isinstance(model, KermtDualCausalCv4)

    for batch_seqs in dataloader:
        unique_smiles = list({s["smiles"] for s in batch_seqs})
        smi_to_idx = {s: i for i, s in enumerate(unique_smiles)}

        graph_batch = _build_graph_batch(unique_smiles, shared_dict, kermt_args)
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
            all_true.extend(rf_true.cpu().tolist())
            all_pred.extend(rf_pred.cpu().tolist())

    return np.array(all_true, dtype=np.float64), np.array(all_pred, dtype=np.float64)


@torch.no_grad()
def predict_split_c_v4_d2(model, dataloader, device, shared_dict, kermt_args):
    """C v4 D2：与 v3 相同逐步前向（ensemble），无 predict_sequence。"""
    from c_v4_d2_transformer import KermtCv4D2Transformer

    model.eval()
    all_true, all_pred = [], []
    assert isinstance(model, KermtCv4D2Transformer)

    for batch_seqs in dataloader:
        unique_smiles = list({s["smiles"] for s in batch_seqs})
        smi_to_idx = {s: i for i, s in enumerate(unique_smiles)}

        graph_batch = _build_graph_batch(unique_smiles, shared_dict, kermt_args)
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
            if int(getattr(model, "num_channels", 0) or 0) > 0:
                from c_v4_d2_transformer import align_steps_to_k

                K = int(model.num_channels)
                rf_k, _, _, valid = align_steps_to_k(rf_true, solvent_seq, desc_seq, K)
                vm = valid.cpu().numpy().astype(bool)
                all_true.extend(rf_k.cpu().numpy()[vm].tolist())
                all_pred.extend(pred.cpu().numpy()[vm].tolist())
            else:
                all_true.extend(rf_true.cpu().tolist())
                all_pred.extend(pred.cpu().tolist())

    return np.array(all_true, dtype=np.float64), np.array(all_pred, dtype=np.float64)


def _load_cfg(model_dir: str):
    cfg_path = os.path.join(model_dir, "effective_config.yaml")
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(model_dir, "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f), cfg_path


def run_eval_and_plot(
    plan: str,
    model_dir: str,
    output_dir: str | None = None,
    batch_size: int | None = None,
    gpu: int | None = None,
    title_prefix: str | None = None,
    suptitle: str | None = None,
) -> str:
    """
    运行评测并绘图。返回 output_dir 绝对路径。
    """
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(KERMT_ROOT, model_dir)

    out = output_dir or os.path.join(model_dir, "plots")
    if not os.path.isabs(out):
        out = os.path.join(KERMT_ROOT, out)
    os.makedirs(out, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg, _ = _load_cfg(model_dir)

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(KERMT_ROOT, p)

    train_seq = _abs(cfg["train_seq"])
    val_seq = _abs(cfg["val_seq"])
    test_seq = _abs(cfg["test_seq"])
    checkpoint_pretrained = _abs(cfg["checkpoint"])

    if gpu is not None:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    kermt_args = Namespace(
        cuda=(device.type == "cuda"),
        dropout=0.0,
        activation="PReLU",
        self_attention=False,
        attn_hidden=4,
        attn_out=128,
        bond_drop_rate=float(cfg.get("bond_drop_rate", 0.0)),
        no_cache=True,
        use_cuikmolmaker_featurization=False,
    )

    if plan == "autoregressive":
        if batch_size is None:
            batch_size = 64
        tp = title_prefix or "Plan B — Autoregressive"
        st = suptitle or f"{tp} — Parity Plots"
    elif plan == "bidirectional_c":
        if batch_size is None:
            batch_size = int(cfg.get("batch_size", 32))
        tp = title_prefix or "Plan C — Bidirectional"
        st = suptitle or f"{tp} — Parity Plots"
    elif plan == "c_v3":
        if batch_size is None:
            batch_size = int(cfg.get("batch_size", 128))
        tp = title_prefix or "Plan C v3 — Vector regression"
        st = suptitle or f"{tp} — Parity Plots"
    elif plan == "c_v4":
        if batch_size is None:
            batch_size = int(cfg.get("batch_size", 128))
        tp = title_prefix or "Plan C v4 — Dual causal + ensemble"
        st = suptitle or f"{tp} — Parity Plots"
    elif plan == "c_v4_d2":
        if batch_size is None:
            batch_size = int(cfg.get("batch_size", 128))
        tp = title_prefix or "Plan C v4 D2 — Causal Transformer dual-tower"
        st = suptitle or f"{tp} — Parity Plots"
    else:
        raise ValueError(f"Unknown plan: {plan!r}")

    print("Loading KERMT encoder from pretrained checkpoint...")
    encoder, readout, mol_dim = load_kermt_encoder(checkpoint_pretrained, kermt_args)

    best_ckpt = os.path.join(model_dir, "best_model.pt")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]

    if plan == "autoregressive":
        use_legacy = any(k.startswith("solvent_mlp.") for k in sd)
        if use_legacy:
            print("  Checkpoint uses legacy SolventMLP — building KermtAutoregressiveLegacy")
            model = KermtAutoregressiveLegacy(
                kermt_encoder=encoder,
                readout=readout,
                mol_dim=mol_dim,
                graph_args=kermt_args,
                desc_dim=cfg.get("desc_dim", 6),
                desc_emb_dim=cfg.get("desc_emb_dim", 64),
                ffn_hidden=cfg.get("ffn_hidden", 512),
                ffn_blocks=cfg.get("ffn_blocks", 3),
                dropout=cfg.get("dropout", 0.1),
            ).to(device)
        else:
            model = KermtAutoregressive(
                kermt_encoder=encoder,
                readout=readout,
                mol_dim=mol_dim,
                graph_args=kermt_args,
                desc_dim=cfg.get("desc_dim", 6),
                desc_emb_dim=cfg.get("desc_emb_dim", 64),
                n_solvents=cfg.get("n_solvents", 5),
                ffn_hidden=cfg.get("ffn_hidden", 512),
                ffn_blocks=cfg.get("ffn_blocks", 3),
                dropout=cfg.get("dropout", 0.1),
            ).to(device)
        print(f"Loading best model weights from {best_ckpt}")
        if use_legacy:
            miss, _ = model.load_state_dict(sd, strict=False)
            if miss:
                print(f"  (strict=False) missing {len(miss)} keys")
        else:
            model.load_state_dict(sd)
    elif plan == "bidirectional_c":
        from bidirectional_c_model import KermtBidirectionalCv1

        model = KermtBidirectionalCv1(
            kermt_encoder=encoder,
            readout=readout,
            mol_dim=mol_dim,
            graph_args=kermt_args,
            desc_dim=int(cfg.get("desc_dim", 6)),
            n_solvents=int(cfg.get("n_solvents", 5)),
            mlp_hidden=int(cfg.get("mlp_hidden", 512)),
            mlp_blocks=int(cfg.get("mlp_blocks", 3)),
            dropout=float(cfg.get("dropout", 0.1)),
        ).to(device)
        print(f"Loading best model from {best_ckpt}")
        model.load_state_dict(sd)
    elif plan in ("c_v3", "c_v4"):
        from c_v3_c_v4_model import KermtDualCausalCv4, KermtVectorCv3

        if plan == "c_v3":
            model = KermtVectorCv3(
                kermt_encoder=encoder,
                readout=readout,
                mol_dim=mol_dim,
                graph_args=kermt_args,
                desc_dim=int(cfg.get("desc_dim", 6)),
                n_solvents=int(cfg.get("n_solvents", 5)),
                mlp_hidden=int(cfg.get("mlp_hidden", 512)),
                mlp_blocks=int(cfg.get("mlp_blocks", 3)),
                dropout=float(cfg.get("dropout", 0.1)),
            ).to(device)
        else:
            model = KermtDualCausalCv4(
                kermt_encoder=encoder,
                readout=readout,
                mol_dim=mol_dim,
                graph_args=kermt_args,
                desc_dim=int(cfg.get("desc_dim", 6)),
                n_solvents=int(cfg.get("n_solvents", 5)),
                mlp_hidden=int(cfg.get("mlp_hidden", 512)),
                mlp_blocks=int(cfg.get("mlp_blocks", 3)),
                dropout=float(cfg.get("dropout", 0.1)),
            ).to(device)
        print(f"Loading best model from {best_ckpt} (plan={plan})")
        model.load_state_dict(sd)
    elif plan == "c_v4_d2":
        from c_v4_d2_transformer import KermtCv4D2Transformer

        tdf = cfg.get("transformer_dim_ff", 0)
        dim_ff = int(tdf) if tdf else None
        if dim_ff == 0:
            dim_ff = None
        model = KermtCv4D2Transformer(
            kermt_encoder=encoder,
            readout=readout,
            mol_dim=mol_dim,
            graph_args=kermt_args,
            desc_dim=int(cfg.get("desc_dim", 6)),
            n_solvents=int(cfg.get("n_solvents", 5)),
            d_model=int(cfg.get("transformer_d_model", 256)),
            nhead=int(cfg.get("transformer_nhead", 4)),
            num_layers=int(cfg.get("transformer_nlayers", 2)),
            dim_feedforward=dim_ff,
            dropout=float(cfg.get("dropout", 0.1)),
            num_channels=int(cfg.get("num_channels", 0)),
        ).to(device)
        print(f"Loading best model from {best_ckpt} (plan=c_v4_d2)")
        model.load_state_dict(sd)
    else:
        raise ValueError(f"Unhandled plan for model build: {plan!r}")

    best_epoch = ckpt.get("epoch", "?")
    best_mae = ckpt.get("val_mae", "?")
    print(f"  Best epoch: {best_epoch}, best val MAE: {best_mae}")

    model.eval()
    shared_dict = {}

    print("\n" + "=" * 60)
    print(f"  {tp} — Evaluate & Plot")
    print("=" * 60)

    log_path = os.path.join(model_dir, "train.log")
    if os.path.isfile(log_path):
        print("\n[1/4] Plotting training curves...")
        if plan == "autoregressive":
            plot_training_curves_beta(log_path, out, timestamp)
        elif plan in ("c_v3", "c_v4", "c_v4_d2"):
            plot_training_curves_c_v3_v4(log_path, out, timestamp)
        else:
            plot_training_curves_scheme_c(log_path, out, timestamp)
    else:
        print("\n[1/4] No train.log found, skipping training curves.")

    splits = {
        "Train": train_seq,
        "Valid": val_seq,
        "Test": test_seq,
    }

    all_metrics = {}
    step = 2
    eval_refine = int(cfg.get("eval_refine", 5)) if plan == "bidirectional_c" else 0

    for split_name, seq_path in splits.items():
        print(f"\n[{step}/4] Evaluating {split_name} set...")
        if not os.path.isfile(seq_path):
            print(f"  Warning: {seq_path} not found, skipping.")
            step += 1
            continue

        ds = SequenceDataset(seq_path)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate,
            num_workers=0,
        )

        t0 = time.time()
        if plan == "autoregressive":
            y_true, y_pred = predict_split_autoregressive(
                model, loader, device, shared_dict, kermt_args
            )
        elif plan == "bidirectional_c":
            y_true, y_pred = predict_split_bidirectional(
                model, loader, device, shared_dict, kermt_args, eval_refine
            )
        elif plan == "c_v3":
            y_true, y_pred = predict_split_c_v3(
                model, loader, device, shared_dict, kermt_args
            )
        elif plan == "c_v4":
            y_true, y_pred = predict_split_c_v4(
                model, loader, device, shared_dict, kermt_args
            )
        elif plan == "c_v4_d2":
            y_true, y_pred = predict_split_c_v4_d2(
                model, loader, device, shared_dict, kermt_args
            )
        else:
            raise ValueError(f"Unknown plan for inference: {plan!r}")
        elapsed = time.time() - t0
        print(f"  Inference: {len(y_true)} steps in {elapsed:.1f}s")

        metrics = plot_parity_split(
            y_true, y_pred, out, split_name, timestamp, title_prefix=tp
        )
        all_metrics[split_name] = metrics

        pred_csv = os.path.join(out, f"{split_name.lower()}_predictions.csv")
        np.savetxt(
            pred_csv,
            np.column_stack([y_true, y_pred]),
            delimiter=",",
            header="true_rf,pred_rf",
            comments="",
        )
        step += 1

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  {'Split':10s} {'R²':>8s} {'MAE':>8s} {'RMSE':>8s} {'n':>8s}")
    print("  " + "-" * 42)
    for name, m in all_metrics.items():
        print(f"  {name:10s} {m['r2']:8.4f} {m['mae']:8.4f} {m['rmse']:8.4f} {m['n']:8d}")
    print("=" * 60)

    plot_combined_parity(all_metrics, out, timestamp, suptitle=st)

    print(f"\nAll outputs saved to: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="TLC sequence models: parity plots (Plan B autoregressive or Plan C bidirectional)"
    )
    parser.add_argument(
        "--plan",
        type=str,
        choices=("autoregressive", "bidirectional_c", "c_v3", "c_v4", "c_v4_d2"),
        default="autoregressive",
        help="autoregressive=B；bidirectional_c=C v1；c_v3/c_v4/c_v4_d2=train_c_v3_v4 模型",
    )
    parser.add_argument("--model_dir", type=str, default="tlc/results/autoregressive")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="默认：B=64，C=读 config batch_size",
    )
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--title_prefix",
        type=str,
        default=None,
        help="覆盖图标题前缀（默认 Plan B / Plan C）",
    )
    parser.add_argument(
        "--suptitle",
        type=str,
        default=None,
        help="覆盖 parity_combined 总标题",
    )
    args = parser.parse_args()

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
