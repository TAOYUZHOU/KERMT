"""
方案 C v3：极简向量回归 — 单头逐步输出 R_f，无 prev/next 邻域、无步间正则。

方案 C v4：双因果解码（前向 + 反向）+ ensemble，无步间 relu 正则；
训练时 teacher forcing；推理时逐步自回归，再对两路取均值。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERMT_ROOT = str(Path(__file__).resolve().parents[2])
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_KERMT_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kermt.model.models import SOLVENT_SMILES


def _make_trunk(in_dim: int, hidden: int, n_blocks: int, dropout: float) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
    for _ in range(max(0, n_blocks - 1)):
        layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class _TLCSeqBase(nn.Module):
    """共享：分子 GNN、溶剂向量、逐步融合特征 x[t]。"""

    def __init__(
        self,
        kermt_encoder,
        readout,
        mol_dim: int,
        graph_args,
        desc_dim: int = 6,
        n_solvents: int = 5,
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
        a_scope_list = (
            a_scope.data.cpu().numpy().tolist() if isinstance(a_scope, torch.Tensor) else a_scope
        )
        return self.encode_molecules(graph, a_scope_list)

    def _step_features(self, mol_vec, solvent_seq, desc_seq, solvent_vecs):
        """mol_vec [mol_dim]; solvent_seq [T,5]; desc_seq [T,6]; solvent_vecs [5,mol_dim] -> [T,fuse_dim]"""
        T = solvent_seq.size(0)
        mol_exp = mol_vec.unsqueeze(0).expand(T, -1)
        mol_cat = torch.cat([mol_exp, desc_seq], dim=-1)
        ratios = solvent_seq.unsqueeze(-1)
        scaled = ratios * solvent_vecs.unsqueeze(0)
        desc_exp = desc_seq.unsqueeze(1).expand(-1, self.n_solvents, -1)
        sol_block = torch.cat([scaled, desc_exp], dim=-1)
        sol_flat = sol_block.reshape(T, -1)
        return torch.cat([mol_cat, sol_flat], dim=-1)


class KermtVectorCv3(_TLCSeqBase):
    """C v3：单 MLP 头，对每步 x[t] 输出标量 R_f（参数在步间共享）。"""

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
        super().__init__(kermt_encoder, readout, mol_dim, graph_args, desc_dim, n_solvents)
        self.head = nn.Sequential(
            _make_trunk(self.fuse_dim, mlp_hidden, mlp_blocks, dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, mol_vec, solvent_seq, desc_seq, solvent_vecs):
        x = self._step_features(mol_vec, solvent_seq, desc_seq, solvent_vecs)
        out = self.head(x).squeeze(-1)
        return out.clamp(1e-4, 1.0 - 1e-4)


class KermtDualCausalCv4(_TLCSeqBase):
    """C v4：前向因果（看 prev_rf）与反向因果（看 next_rf）两支，训练 teacher forcing。"""

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
        super().__init__(kermt_encoder, readout, mol_dim, graph_args, desc_dim, n_solvents)
        d_in = self.fuse_dim + 1
        self.trunk_fwd = _make_trunk(d_in, mlp_hidden, mlp_blocks, dropout)
        self.out_fwd = nn.Linear(mlp_hidden, 1)
        self.trunk_bwd = _make_trunk(d_in, mlp_hidden, mlp_blocks, dropout)
        self.out_bwd = nn.Linear(mlp_hidden, 1)

    def forward_train(
        self,
        mol_vec,
        solvent_seq,
        desc_seq,
        rf_true,
        solvent_vecs,
        boundary_rf: float = 0.5,
    ):
        """Teacher forcing：前向用 rf_true[t-1]；反向用 rf_true[t+1]。

        边界必须与 `predict_sequence(..., init=boundary_rf)` 一致：
        - 前向 t=0 以前一步 Rf 为 **boundary_rf**（推理时与 init 相同），**不能**用 0，否则 train/eval 分布不一致。
        - 反向 t=T-1 的「下一拍」在物理上不存在，应用 **boundary_rf**；若用 rf_true[t] 会引入标签泄漏且与推理不一致。
        """
        x = self._step_features(mol_vec, solvent_seq, desc_seq, solvent_vecs)
        T = rf_true.numel()
        device = rf_true.device
        dtype = x.dtype
        b = torch.tensor(boundary_rf, device=device, dtype=dtype).reshape(1)

        rf_fwd = torch.empty(T, device=device)
        for t in range(T):
            if t > 0:
                prev = rf_true[t - 1].detach().unsqueeze(0)
            else:
                prev = b
            inp = torch.cat([x[t], prev], dim=-1)
            rf_fwd[t] = self.out_fwd(self.trunk_fwd(inp)).squeeze()

        rf_bwd = torch.empty(T, device=device)
        for t in range(T - 1, -1, -1):
            if t == T - 1:
                nxt = b
            else:
                nxt = rf_true[t + 1].detach().unsqueeze(0)
            inp = torch.cat([x[t], nxt], dim=-1)
            rf_bwd[t] = self.out_bwd(self.trunk_bwd(inp)).squeeze()

        rf_fwd = rf_fwd.clamp(1e-4, 1.0 - 1e-4)
        rf_bwd = rf_bwd.clamp(1e-4, 1.0 - 1e-4)
        return rf_fwd, rf_bwd

    @torch.no_grad()
    def predict_sequence(
        self, mol_vec, solvent_seq, desc_seq, solvent_vecs, init: float = 0.5
    ):
        """推理：两路自回归后 ensemble。"""
        x = self._step_features(mol_vec, solvent_seq, desc_seq, solvent_vecs)
        T = x.size(0)
        device = x.device
        z = torch.tensor(init, device=device)

        rf_fwd = torch.empty(T, device=device)
        for t in range(T):
            prev = rf_fwd[t - 1] if t > 0 else z
            inp = torch.cat([x[t], prev.unsqueeze(0)], dim=-1)
            rf_fwd[t] = self.out_fwd(self.trunk_fwd(inp)).squeeze().clamp(1e-4, 1.0 - 1e-4)

        rf_bwd = torch.empty(T, device=device)
        for t in range(T - 1, -1, -1):
            if t == T - 1:
                nxt = z
            else:
                nxt = rf_bwd[t + 1]
            inp = torch.cat([x[t], nxt.unsqueeze(0)], dim=-1)
            rf_bwd[t] = self.out_bwd(self.trunk_bwd(inp)).squeeze().clamp(1e-4, 1.0 - 1e-4)

        rf = 0.5 * (rf_fwd + rf_bwd)
        return rf, rf_fwd, rf_bwd


def regression_loss_tensor(pred, target, kind: str) -> torch.Tensor:
    import torch.nn.functional as F

    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "mae":
        return F.l1_loss(pred, target)
    raise ValueError(kind)
