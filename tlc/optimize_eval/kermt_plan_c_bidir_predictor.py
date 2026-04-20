#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KERMT 方案 C（双向序列）在 optimize_tlc 中的先验：predict_batch(smiles, solvent_arrs)->(mu, std)。

对每个 (SMILES, 5 维溶剂向量) 单独做 T=1 的 predict_sequence，避免把 BO 网格上的多个比例当成一条有序序列耦合。

依赖：KERMT 仓库根在 kermt_root；save_dir 含 best_model.pt 与 effective_config.yaml；
     描述符 z-score 默认 tlc/data/descriptor_zscore_stats.json（与序列训练一致）。
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

SOLVENT_TO_IDX = {"H": 0, "EA": 1, "DCM": 2, "MeOH": 3, "Et2O": 4}


def _ensure_paths(kermt_root: Path) -> None:
    root = str(kermt_root.resolve())
    scripts = str((kermt_root / "tlc/scripts").resolve())
    for p in (root, scripts):
        if p not in sys.path:
            sys.path.insert(0, p)


def _rdkit_desc_raw(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return np.array(
        [
            float(Descriptors.MolWt(mol)),
            float(Descriptors.TPSA(mol)),
            float(Descriptors.MolLogP(mol)),
            float(Lipinski.NumHDonors(mol)),
            float(Lipinski.NumHAcceptors(mol)),
            float(Lipinski.NumRotatableBonds(mol)),
        ],
        dtype=np.float64,
    )


def _load_desc_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    mean = np.asarray(data["desc_mean"], dtype=np.float64)
    std = np.asarray(data["desc_std"], dtype=np.float64)
    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError("descriptor stats must be 6-d")
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


class KermtPlanCBidirPredictor:
    """与 KermtPlanAV1Predictor / UniMol 对齐的 optimize 接口。"""

    def __init__(
        self,
        *,
        kermt_root: str | Path,
        bidir_save_dir: str | Path,
        desc_stats_json: str | Path | None = None,
        solvent1: str = "H",
        solvent2: str = "EA",
        min_std: float = 0.02,
        script_dir: str | Path | None = None,
        gpu: int | None = None,
        batch_size: int = 64,
        eval_refine: int | None = None,
    ):
        self.kermt_root = Path(kermt_root).expanduser().resolve()
        self.save_dir = Path(bidir_save_dir).expanduser()
        if not self.save_dir.is_absolute():
            self.save_dir = (self.kermt_root / self.save_dir).resolve()
        if not (self.save_dir / "best_model.pt").is_file():
            raise FileNotFoundError(f"best_model.pt not found under {self.save_dir}")

        if desc_stats_json is None:
            desc_stats_json = self.kermt_root / "tlc/data/descriptor_zscore_stats.json"
        dsp = Path(desc_stats_json).expanduser()
        if not dsp.is_absolute():
            dsp = (self.kermt_root / dsp).resolve()
        if not dsp.is_file():
            raise FileNotFoundError(f"descriptor zscore json missing: {dsp}")
        self._desc_mean, self._desc_std = _load_desc_stats(dsp)

        self.solvent1 = str(solvent1).strip()
        self.solvent2 = str(solvent2).strip()
        self.min_std = float(max(1e-6, min_std))
        self._script_dir = Path(script_dir).resolve() if script_dir else Path(__file__).resolve().parent

        if gpu is None:
            gpu = int(os.environ.get("KERMT_PLAN_C_GPU", "0"))
        self._gpu = int(gpu)
        self._batch_size = int(batch_size)

        cfg_path = self.save_dir / "effective_config.yaml"
        if not cfg_path.is_file():
            cfg_path = self.save_dir / "config.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)
        self._eval_refine = int(
            eval_refine if eval_refine is not None else self._cfg.get("eval_refine", 5)
        )

        self._device: torch.device | None = None
        self._model = None
        self._kermt_args = None
        self._shared_dict: dict | None = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        _ensure_paths(self.kermt_root)
        from argparse import Namespace

        from autoregressive_model import load_kermt_encoder
        from bidirectional_c_model import KermtBidirectionalCv1

        self._device = torch.device(
            f"cuda:{self._gpu}" if torch.cuda.is_available() else "cpu"
        )
        ckpt_path = self.save_dir / "best_model.pt"
        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=False)

        self._kermt_args = Namespace(
            cuda=(self._device.type == "cuda"),
            dropout=0.0,
            activation="PReLU",
            self_attention=False,
            attn_hidden=4,
            attn_out=128,
            bond_drop_rate=0.0,
            no_cache=True,
            use_cuikmolmaker_featurization=False,
        )

        checkpoint_pretrained = ckpt["args"].get("checkpoint", "pretrained_models/grover_base.pt")
        cpp = Path(checkpoint_pretrained).expanduser()
        if not cpp.is_absolute():
            cpp = (self.kermt_root / cpp).resolve()

        encoder, readout, mol_dim = load_kermt_encoder(str(cpp), self._kermt_args)
        model = KermtBidirectionalCv1(
            kermt_encoder=encoder,
            readout=readout,
            mol_dim=mol_dim,
            graph_args=self._kermt_args,
            desc_dim=int(self._cfg.get("desc_dim", 6)),
            n_solvents=int(self._cfg.get("n_solvents", 5)),
            mlp_hidden=int(self._cfg.get("mlp_hidden", 512)),
            mlp_blocks=int(self._cfg.get("mlp_blocks", 3)),
            dropout=float(self._cfg.get("dropout", 0.1)),
        ).to(self._device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        self._model = model
        self._shared_dict = {}
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)

    def _desc_z(self, smiles: str) -> np.ndarray:
        raw = _rdkit_desc_raw(smiles)
        return ((raw - self._desc_mean) / self._desc_std).astype(np.float32)

    @torch.no_grad()
    def _predict_one(self, smiles: str, solvent5: np.ndarray) -> float:
        self._lazy_init()
        assert self._model is not None and self._device is not None and self._shared_dict is not None
        from kermt.data.molgraph import mol2graph

        s5 = np.asarray(solvent5, dtype=np.float32).reshape(5)
        desc = self._desc_z(smiles)

        graph_batch = mol2graph([smiles], self._shared_dict, self._kermt_args)
        gb = graph_batch.get_components()
        _, _, _, _, _, a_scope, _, _ = gb
        a_scope_list = a_scope.data.cpu().numpy().tolist()

        mol_vecs = self._model.encode_molecules(gb, a_scope_list)
        solvent_vecs = self._model.encode_solvents()
        mol_vec = mol_vecs[0]

        solvent_seq = torch.tensor(s5.reshape(1, 5), dtype=torch.float32, device=self._device)
        desc_seq = torch.tensor(desc.reshape(1, 6), dtype=torch.float32, device=self._device)

        rf_pred, _, _ = self._model.predict_sequence(
            mol_vec, solvent_seq, desc_seq, solvent_vecs, n_refine=self._eval_refine
        )
        return float(rf_pred[0].cpu().item())

    def predict_batch(self, smiles: str, solvent_arrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arrs = np.asarray(solvent_arrs, dtype=np.float32)
        if arrs.ndim == 1:
            arrs = arrs.reshape(1, -1)
        n = arrs.shape[0]
        if arrs.shape[1] != 5:
            raise ValueError(f"expected solvent arr (N,5), got {arrs.shape}")
        mus = np.empty(n, dtype=np.float64)
        for i in range(n):
            try:
                mus[i] = self._predict_one(smiles, arrs[i])
            except Exception as e:
                warnings.warn(f"Plan C predict failed row {i}: {e}", RuntimeWarning)
                mus[i] = 0.5
        sig = np.full(n, self.min_std, dtype=np.float64)
        return mus, sig
