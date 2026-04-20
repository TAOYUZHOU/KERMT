"""
方案 B: 自回归单调 Rf 预测模型

架构:
  SMILES → KERMT GNN (unfrozen, full finetune) → Readout → mol_vec [mol_dim]
  5 solvent SMILES → same GNN → 5 × [mol_dim], scaled by ratio, concat [5*mol_dim]
    → 3-layer ResidualMLP → solv_proj [mol_dim]
  desc [6] → Linear(6→64) → desc_emb [64]
  X = [mol_vec ⊕ solv_proj ⊕ desc_emb ⊕ prev_rf] → ResidualFFN → Beta(μ,φ)
"""
import math
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

import sys
from pathlib import Path
_KERMT_ROOT = str(Path(__file__).resolve().parents[2])
if _KERMT_ROOT not in sys.path:
    sys.path.insert(0, _KERMT_ROOT)

from kermt.model.models import SolventProjection, SOLVENT_SMILES


class ResidualFFNBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return self.norm(x + residual)


class BetaRegressionHead(nn.Module):
    """Predict Beta distribution parameters (mu, phi) from input features."""

    def __init__(self, in_dim, hidden_dim=512, n_blocks=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualFFNBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.head_mu = nn.Linear(hidden_dim, 1)
        self.head_log_phi = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.gelu(self.proj(x))
        for block in self.blocks:
            x = block(x)
        mu = torch.sigmoid(self.head_mu(x)).squeeze(-1)
        log_phi = self.head_log_phi(x).squeeze(-1)
        phi = F.softplus(log_phi) + 2.0
        return mu, phi


def beta_nll_loss(mu, phi, target, eps=1e-6):
    target = target.clamp(eps, 1.0 - eps)
    alpha = mu * phi
    beta = (1.0 - mu) * phi

    log_prob = (
        torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + (alpha - 1.0) * torch.log(target)
        + (beta - 1.0) * torch.log(1.0 - target)
    )
    return -log_prob.mean()


class KermtAutoregressive(nn.Module):
    """
    Autoregressive monotonic Rf predictor built on KERMT encoder.

    Solvents are encoded by the same GNN encoder as molecules:
      - 5 solvent SMILES → GNN → 5 vectors of mol_dim
      - Each scaled by its ratio, concatenated, projected via 3-layer residual MLP
    """

    def __init__(self, kermt_encoder, readout, mol_dim, graph_args,
                 desc_dim=6, n_solvents=5, desc_emb_dim=64,
                 ffn_hidden=512, ffn_blocks=3, dropout=0.1):
        super().__init__()
        self.kermt_encoder = kermt_encoder
        self.readout = readout
        self.mol_dim = mol_dim

        self._build_solvent_graph(graph_args)
        self.solvent_proj = SolventProjection(mol_dim, n_solvents)
        self.desc_proj = nn.Linear(desc_dim, desc_emb_dim)

        head_input_dim = 2 * mol_dim + desc_emb_dim + 1
        self.beta_head = BetaRegressionHead(
            head_input_dim, ffn_hidden, ffn_blocks, dropout
        )

    def _build_solvent_graph(self, args):
        """Pre-build molecular graphs for 5 solvents (stored on CPU)."""
        from kermt.data.molgraph import mol2graph
        bg = mol2graph(SOLVENT_SMILES, {}, args)
        self._solvent_graph_cpu = bg.get_components()

    def encode_molecules(self, graph_batch, a_scope):
        """Run KERMT encoder + readout → mol_vecs [B, mol_dim]."""
        output = self.kermt_encoder(graph_batch)

        mol_from_atom = self.readout(output["atom_from_atom"], a_scope)
        mol_from_bond = self.readout(output["atom_from_bond"], a_scope)

        if isinstance(mol_from_atom, tuple):
            mol_from_atom = mol_from_atom[0].flatten(start_dim=1)
            mol_from_bond = mol_from_bond[0].flatten(start_dim=1)

        return (mol_from_atom + mol_from_bond) / 2.0

    def encode_solvents(self):
        """Encode 5 solvent SMILES through the shared encoder → [5, mol_dim]."""
        device = next(self.parameters()).device
        graph = tuple(
            c.to(device) if isinstance(c, torch.Tensor) else c
            for c in self._solvent_graph_cpu
        )
        _, _, _, _, _, a_scope, _, _ = graph
        a_scope_list = a_scope.data.cpu().numpy().tolist() if isinstance(a_scope, torch.Tensor) else a_scope
        return self.encode_molecules(graph, a_scope_list)

    def forward_step(self, mol_vec, solvent_ratios, desc_6, prev_rf, solvent_vecs):
        """
        Single autoregressive step.

        mol_vec:        [B, mol_dim]
        solvent_ratios: [B, 5]
        desc_6:         [B, 6]
        prev_rf:        [B]
        solvent_vecs:   [5, mol_dim]  (pre-encoded, shared across steps)
        """
        solv_proj = self.solvent_proj(solvent_vecs, solvent_ratios)  # [B, mol_dim]
        desc_emb = F.gelu(self.desc_proj(desc_6))
        prev_rf_input = prev_rf.unsqueeze(-1)

        x = torch.cat([mol_vec, solv_proj, desc_emb, prev_rf_input], dim=-1)
        mu, phi = self.beta_head(x)
        return mu, phi

    def predict_sequence(self, mol_vec, solvent_seq, desc_seq,
                         system_boundaries, solvent_vecs,
                         true_rf_seq=None, teacher_forcing_ratio=1.0):
        """
        Predict Rf for a full sequence with monotonic constraints.

        mol_vec:      [mol_dim]
        solvent_seq:  [T, 5]  raw solvent ratios per step
        desc_seq:     [T, 6]
        solvent_vecs: [5, mol_dim]  (pre-encoded, reused across steps)
        """
        T = solvent_seq.size(0)
        device = mol_vec.device
        rf_preds = []
        mus = []
        phis = []

        prev_rf = torch.tensor(-1.0, device=device)
        boundary_set = set(system_boundaries)

        mol_expanded = mol_vec.unsqueeze(0)

        for t in range(T):
            is_system_start = (t in boundary_set)

            mu, phi = self.forward_step(
                mol_expanded,
                solvent_seq[t:t+1],
                desc_seq[t:t+1],
                prev_rf.unsqueeze(0),
                solvent_vecs,
            )
            mu = mu[0]
            phi = phi[0]
            mus.append(mu)
            phis.append(phi)

            if is_system_start or prev_rf < 0:
                rf_t = mu
            else:
                margin = 1.0 - prev_rf
                delta = mu * margin
                rf_t = prev_rf + delta

            rf_preds.append(rf_t)

            if true_rf_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_rf = true_rf_seq[t]
            else:
                prev_rf = rf_t.detach()

        return (
            torch.stack(rf_preds),
            torch.stack(mus),
            torch.stack(phis),
        )


class _LegacySolventMLP(nn.Module):
    """与旧 checkpoint 中 solvent_mlp.net.* 结构一致。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
        )

    def forward(self, x):
        return self.net(x)


class KermtAutoregressiveLegacy(nn.Module):
    """
    旧版方案 B：溶剂用 5 维比例 + 10 维交叉项 (共 15 维) 经 MLP 融合，不经 GNN 编码溶剂。
    与早期 `best_model.pt`（含 solvent_mlp.net.*）权重兼容。
    """

    def __init__(self, kermt_encoder, readout, mol_dim, graph_args,
                 desc_dim=6, desc_emb_dim=64,
                 ffn_hidden=512, ffn_blocks=3, dropout=0.1):
        super().__init__()
        self.kermt_encoder = kermt_encoder
        self.readout = readout
        self.mol_dim = mol_dim

        self.solvent_mlp = _LegacySolventMLP()
        self.desc_proj = nn.Linear(desc_dim, desc_emb_dim)

        head_input_dim = mol_dim + 512 + desc_emb_dim + 1
        self.beta_head = BetaRegressionHead(
            head_input_dim, ffn_hidden, ffn_blocks, dropout
        )

    def encode_molecules(self, graph_batch, a_scope):
        output = self.kermt_encoder(graph_batch)

        mol_from_atom = self.readout(output["atom_from_atom"], a_scope)
        mol_from_bond = self.readout(output["atom_from_bond"], a_scope)

        if isinstance(mol_from_atom, tuple):
            mol_from_atom = mol_from_atom[0].flatten(start_dim=1)
            mol_from_bond = mol_from_bond[0].flatten(start_dim=1)

        return (mol_from_atom + mol_from_bond) / 2.0

    def forward_step(self, mol_vec, solvent_cross_15, desc_6, prev_rf):
        """solvent_cross_15: [B, 15] = concat(溶剂5, 交叉10)"""
        solv_proj = self.solvent_mlp(solvent_cross_15)
        desc_emb = F.gelu(self.desc_proj(desc_6))
        prev_rf_input = prev_rf.unsqueeze(-1)
        x = torch.cat([mol_vec, solv_proj, desc_emb, prev_rf_input], dim=-1)
        mu, phi = self.beta_head(x)
        return mu, phi

    def predict_sequence(
        self,
        mol_vec,
        solvent_seq,
        cross_seq,
        desc_seq,
        system_boundaries,
        true_rf_seq=None,
        teacher_forcing_ratio=1.0,
    ):
        """
        solvent_seq: [T, 5], cross_seq: [T, 10], desc_seq: [T, 6]
        """
        T = solvent_seq.size(0)
        device = mol_vec.device
        feat15 = torch.cat([solvent_seq, cross_seq], dim=-1)

        rf_preds = []
        mus = []
        phis = []

        prev_rf = torch.tensor(-1.0, device=device)
        boundary_set = set(system_boundaries)
        mol_expanded = mol_vec.unsqueeze(0)

        for t in range(T):
            is_system_start = (t in boundary_set)

            mu, phi = self.forward_step(
                mol_expanded,
                feat15[t : t + 1],
                desc_seq[t : t + 1],
                prev_rf.unsqueeze(0),
            )
            mu = mu[0]
            phi = phi[0]
            mus.append(mu)
            phis.append(phi)

            if is_system_start or prev_rf < 0:
                rf_t = mu
            else:
                margin = 1.0 - prev_rf
                delta = mu * margin
                rf_t = prev_rf + delta

            rf_preds.append(rf_t)

            if true_rf_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_rf = true_rf_seq[t]
            else:
                prev_rf = rf_t.detach()

        return (
            torch.stack(rf_preds),
            torch.stack(mus),
            torch.stack(phis),
        )


def load_kermt_encoder(checkpoint_path, args):
    """Load pretrained KERMT encoder + readout from checkpoint."""
    from kermt.model.models import KERMTEmbedding
    from kermt.model.layers import Readout

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loaded_args = state.get("args", state.get("data_args", None))

    if loaded_args is not None:
        from kermt.util.utils import get_model_args
        for key in get_model_args():
            if hasattr(loaded_args, key):
                setattr(args, key, getattr(loaded_args, key))

    encoder = KERMTEmbedding(args)

    if args.self_attention:
        readout = Readout(
            rtype="self_attention",
            hidden_size=args.hidden_size,
            attn_hidden=args.attn_hidden,
            attn_out=args.attn_out,
        )
    else:
        readout = Readout(rtype="mean", hidden_size=args.hidden_size)

    loaded_state = state.get("state_dict", state)
    loaded_state = {k.replace("grover", "kermt"): v for k, v in loaded_state.items()}

    encoder_state = {}
    readout_state = {}
    for k, v in loaded_state.items():
        if k.startswith("kermt."):
            encoder_state[k[len("kermt."):]] = v
        elif k.startswith("readout."):
            readout_state[k[len("readout."):]] = v

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    print(f"  Encoder: loaded {len(encoder_state)} params, "
          f"missing {len(missing)}, unexpected {len(unexpected)}")
    if readout_state:
        readout.load_state_dict(readout_state, strict=False)
        print(f"  Readout: loaded {len(readout_state)} params")
    else:
        print("  Readout: no pretrained weights (using random init)")

    mol_dim = args.hidden_size
    if args.self_attention:
        mol_dim = args.hidden_size * args.attn_out

    return encoder, readout, mol_dim
