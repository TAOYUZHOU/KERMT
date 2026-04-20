"""
方案 C v1：双向上下文 + 原始 6 维描述符 + Beta 头 + 步间 relu 惩罚。

- 分子：concat(mol_GNN [mol_dim], desc_6) → mol_dim+6
- 溶剂：每步 5×concat(ratio_i·emb_i, desc_6) → 5·(mol_dim+6)
- 前向支路 MLP：concat(融合特征, prev_rf_teacher) → mol_dim+6
- 反向支路 MLP：concat(融合特征, next_rf_teacher) → mol_dim+6
- Beta 头：concat(h_fwd, h_rev) → μ, φ（逐步）
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_KERMT_ROOT = str(Path(__file__).resolve().parents[2])
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_KERMT_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autoregressive_model import BetaRegressionHead, beta_nll_loss
from kermt.model.models import SOLVENT_SMILES


class KermtBidirectionalCv1(nn.Module):
    def __init__(
        self,
        kermt_encoder,
        readout,
        mol_dim: int,
        graph_args,
        desc_dim: int = 6,
        n_solvents: int = 5,
        mlp_hidden: int = 512,
        mlp_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.kermt_encoder = kermt_encoder
        self.readout = readout
        self.mol_dim = mol_dim
        self.desc_dim = desc_dim
        self.n_solvents = n_solvents

        self.d_block = mol_dim + desc_dim
        self.fuse_dim = self.d_block + n_solvents * self.d_block

        self._build_solvent_graph(graph_args)

        self.mlp_fwd = self._make_trunk(self.fuse_dim + 1, mlp_hidden, mlp_blocks, dropout)
        self.mlp_rev = self._make_trunk(self.fuse_dim + 1, mlp_hidden, mlp_blocks, dropout)
        self.proj_fwd = nn.Linear(mlp_hidden, self.d_block)
        self.proj_rev = nn.Linear(mlp_hidden, self.d_block)

        self.beta_head = BetaRegressionHead(
            2 * self.d_block, mlp_hidden, mlp_blocks, dropout
        )

    def _make_trunk(self, in_dim, hidden, n_blocks, dropout):
        layers = [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, n_blocks - 1)):
            layers += [
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        return nn.Sequential(*layers)

    def _build_solvent_graph(self, args):
        from kermt.data.molgraph import mol2graph

        bg = mol2graph(SOLVENT_SMILES, {}, args)
        self._solvent_graph_cpu = bg.get_components()

    def encode_molecules(self, graph_batch, a_scope):
        output = self.kermt_encoder(graph_batch)
        mol_from_atom = self.readout(output["atom_from_atom"], a_scope)
        mol_from_bond = self.readout(output["atom_from_bond"], a_scope)
        if isinstance(mol_from_atom, tuple):
            mol_from_atom = mol_from_atom[0].flatten(start_dim=1)
            mol_from_bond = mol_from_bond[0].flatten(start_dim=1)
        return (mol_from_atom + mol_from_bond) / 2.0

    def encode_solvents(self):
        device = next(self.parameters()).device
        graph = tuple(
            c.to(device) if isinstance(c, torch.Tensor) else c
            for c in self._solvent_graph_cpu
        )
        _, _, _, _, _, a_scope, _, _ = graph
        a_scope_list = a_scope.data.cpu().numpy().tolist() if isinstance(a_scope, torch.Tensor) else a_scope
        return self.encode_molecules(graph, a_scope_list)

    def _step_features(self, mol_vec, solvent_seq, desc_seq, solvent_vecs):
        """mol_vec [mol_dim]; solvent_seq [T,5]; desc_seq [T,6]; solvent_vecs [5,mol_dim]"""
        T = solvent_seq.size(0)
        device = mol_vec.device
        mol_exp = mol_vec.unsqueeze(0).expand(T, -1)
        mol_cat = torch.cat([mol_exp, desc_seq], dim=-1)

        ratios = solvent_seq.unsqueeze(-1)
        scaled = ratios * solvent_vecs.unsqueeze(0)
        desc_exp = desc_seq.unsqueeze(1).expand(-1, self.n_solvents, -1)
        sol_block = torch.cat([scaled, desc_exp], dim=-1)
        sol_flat = sol_block.reshape(T, -1)

        return torch.cat([mol_cat, sol_flat], dim=-1)

    def forward_train(self, mol_vec, solvent_seq, desc_seq, rf_true, solvent_vecs):
        """
        Teacher-forced prev/next rf for bidirectional MLP branches.
        rf_true: [T]
        Returns mus, phis [T], aux dict for logging penalties.
        """
        T = rf_true.numel()
        device = mol_vec.device
        x = self._step_features(mol_vec, solvent_seq, desc_seq, solvent_vecs)

        prev_rf = torch.zeros(T, device=device)
        if T > 1:
            prev_rf[1:] = rf_true[:-1].detach()
        next_rf = torch.zeros(T, device=device)
        if T > 1:
            next_rf[:-1] = rf_true[1:].detach()
        if T > 0:
            next_rf[-1] = rf_true[-1].detach()

        h_f = self.proj_fwd(self.mlp_fwd(torch.cat([x, prev_rf.unsqueeze(-1)], dim=-1)))
        h_r = self.proj_rev(self.mlp_rev(torch.cat([x, next_rf.unsqueeze(-1)], dim=-1)))

        h = torch.cat([h_f, h_r], dim=-1)
        mu, phi = self.beta_head(h)
        return mu, phi

    @torch.no_grad()
    def predict_sequence(
        self, mol_vec, solvent_seq, desc_seq, solvent_vecs, n_refine: int = 5
    ):
        """
        迭代细化：用当前 rf 估计构造 prev/next 上下文，多轮更新 μ（推理用，非严格 EM）。
        返回 (rf_preds, mus, phis) 与方案 B evaluate 接口对齐。
        """
        T = solvent_seq.size(0)
        device = mol_vec.device
        x = self._step_features(mol_vec, solvent_seq, desc_seq, solvent_vecs)
        rf = torch.full((T,), 0.5, device=device)
        mu = rf.clone()
        phi = torch.ones(T, device=device)
        for _ in range(n_refine):
            prev = torch.cat([torch.zeros(1, device=device), rf[:-1]])
            nxt = torch.cat([rf[1:], rf[-1:]])
            h_f = self.proj_fwd(self.mlp_fwd(torch.cat([x, prev.unsqueeze(-1)], dim=-1)))
            h_r = self.proj_rev(self.mlp_rev(torch.cat([x, nxt.unsqueeze(-1)], dim=-1)))
            mu, phi = self.beta_head(torch.cat([h_f, h_r], dim=-1))
            rf = mu.clamp(1e-4, 1.0 - 1e-4)
        return rf, mu, phi


def scheme_c_loss(
    mu,
    phi,
    rf_true,
    lambda_fwd: float,
    lambda_rev: float,
    regression_loss: str = "beta_nll",
):
    """
    主损失 + 可选步间单调正则（λ_fwd / λ_rev；可为 0）。

    regression_loss:
      - ``beta_nll``: Beta 负对数似然（默认）
      - ``mse``: 逐步 MSE(μ, rf_true)
      - ``mae``: 逐步 MAE(|μ − rf_true|)
    """
    rf_hat = mu
    T = rf_hat.numel()

    if regression_loss == "beta_nll":
        main = beta_nll_loss(mu, phi, rf_true)
    elif regression_loss == "mse":
        main = F.mse_loss(mu, rf_true)
    elif regression_loss == "mae":
        main = F.l1_loss(mu, rf_true)
    else:
        raise ValueError(f"Unknown regression_loss: {regression_loss!r}")

    if T <= 1:
        return main, main.detach(), torch.tensor(0.0, device=mu.device), torch.tensor(
            0.0, device=mu.device
        )

    d_inc = rf_hat[1:] - rf_hat[:-1]
    l_fwd = F.relu(-d_inc).mean()

    rv = rf_hat.flip(0)
    d_rev = rv[1:] - rv[:-1]
    l_rev = F.relu(d_rev).mean()

    total = main + lambda_fwd * l_fwd + lambda_rev * l_rev
    return total, main.detach(), l_fwd.detach(), l_rev.detach()
