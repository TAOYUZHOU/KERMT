"""
C v4（目标）· 方案 D2：因果自注意力双塔 + 共享 Transformer trunk。

**序列轴语义（与 v3 多任务 Rf 向量对齐）**

- 与 v3 相同：每位置用 `_step_features` 得到 fuse 向量（分子编码 + 该「通道」下的溶剂比例与描述子）。
- **num_channels = K > 0**：目标输出为 **长度 K 的多任务 Rf 向量**。数据轨迹长为 T：若 **T≥K** 则截断为前 K 步，Transformer 序列长 **K**；若 **T<K** 则在 **长度 T** 上跑 Transformer（不在尾部补零再进 trunk，避免反向塔 flip 后因果位仅见 pad 而 NaN），再将预测 **零填充** 到 K，损失对前 T 维监督。
- **num_channels = 0**（默认）：保留旧行为，序列长度 **T = len(steps)**（随样本变化）。

**双塔**

- 前向塔：沿该序列轴做 **因果 self-attention**；反向塔：**flip → 因果 attention → flip**。
- 共享 `nn.TransformerEncoder`；读出为 `readout_fwd` / `readout_bwd` 两套 `Linear(d_model→1)`；
  输出为两塔逐位置 Rf 的 **均值**，再 clamp。

不依赖 v4 MLP 的 teacher-forcing 解码。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_KERMT_ROOT = str(Path(__file__).resolve().parents[2])
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
for _p in (_KERMT_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from c_v3_c_v4_model import _TLCSeqBase


def align_steps_to_k(
    rf_true: torch.Tensor,
    solvent_seq: torch.Tensor,
    desc_seq: torch.Tensor,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将长度 T 的轨迹对齐到固定 K（标签 + 输入）。

    - T >= K：截断为前 K 步；valid 全 True。
    - T < K：rf 尾部填 0；溶剂/描述子零填充；valid 仅前 T 为 True。

    返回 (rf_k [K], sol_k, desc_k, valid [K])。
    """
    T = solvent_seq.size(0)
    device = solvent_seq.device
    dtype_f = solvent_seq.dtype
    dtype_rf = rf_true.dtype
    ns = solvent_seq.size(1)
    if T >= K:
        return (
            rf_true[:K],
            solvent_seq[:K],
            desc_seq[:K],
            torch.ones(K, device=device, dtype=torch.bool),
        )
    rf_k = torch.zeros(K, device=device, dtype=dtype_rf)
    rf_k[:T] = rf_true
    sol_k = torch.zeros(K, ns, device=device, dtype=dtype_f)
    sol_k[:T] = solvent_seq
    d_k = torch.zeros(K, desc_seq.size(1), device=device, dtype=dtype_f)
    d_k[:T] = desc_seq
    valid = torch.zeros(K, device=device, dtype=torch.bool)
    valid[:T] = True
    return rf_k, sol_k, d_k, valid


class SinusoidalPositionalEncoding(nn.Module):
    """[T, d_model] 加性位置编码。"""

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, d_model] or [1, T, d_model]
        pe = self.pe.to(dtype=x.dtype)
        if x.dim() == 2:
            T, _d = x.shape
            x = x + pe[0, :T, :]
            return self.dropout(x)
        T = x.size(1)
        return self.dropout(x + pe[:, :T, :])


class KermtCv4D2Transformer(_TLCSeqBase):
    """
    共享 TransformerEncoder（因果 mask）×2：原序一次、翻转序一次；读出后 ensemble。

    num_channels：>0 时序列轴为固定 K；0 时序列轴为 len(steps)。
    """

    def __init__(
        self,
        kermt_encoder,
        readout,
        mol_dim: int,
        graph_args,
        desc_dim: int = 6,
        n_solvents: int = 5,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        max_len: int = 8192,
        num_channels: int = 0,
    ):
        super().__init__(kermt_encoder, readout, mol_dim, graph_args, desc_dim, n_solvents)
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.d_model = d_model
        self.num_channels = int(num_channels)
        pe_len = max(max_len, self.num_channels if self.num_channels > 0 else 0, 4096)
        self.step_proj = nn.Linear(self.fuse_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=pe_len, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.readout_fwd = nn.Linear(d_model, 1)
        self.readout_bwd = nn.Linear(d_model, 1)

    def _encode_steps(self, h: torch.Tensor) -> torch.Tensor:
        """h: [L, d_model] -> Transformer -> [L, d_model] 因果自注意力（无序列内 padding）。"""
        L = h.size(0)
        h2 = h.unsqueeze(0)  # [1, L, d_model]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=h.device, dtype=h.dtype)
        try:
            out = self.trunk(h2, mask=mask, is_causal=True)
        except TypeError:
            out = self.trunk(h2, mask=mask)
        return out.squeeze(0)

    def _forward_from_x(self, x: torch.Tensor) -> torch.Tensor:
        """x: [L, fuse_dim]，L=T 或 L=K；返回 [L] 的 ensemble Rf。"""
        h = self.step_proj(x)
        h = self.pos_enc(h)
        h_fwd = self._encode_steps(h)
        rf_fwd = self.readout_fwd(h_fwd).squeeze(-1).clamp(1e-4, 1.0 - 1e-4)
        h_bwd_in = h.flip(0)
        h_bwd = self._encode_steps(h_bwd_in).flip(0)
        rf_bwd = self.readout_bwd(h_bwd).squeeze(-1).clamp(1e-4, 1.0 - 1e-4)
        return 0.5 * (rf_fwd + rf_bwd)

    def forward(self, mol_vec, solvent_seq, desc_seq, solvent_vecs):
        """
        Returns:
            rf_ens: 可变 len(steps) 时为 [T]；num_channels=K>0 时恒为 [K]（T<K 时尾部为 0，由损失 mask 忽略）。
        """
        if self.num_channels and self.num_channels > 0:
            K = self.num_channels
            T = solvent_seq.size(0)
            if T >= K:
                sol_k = solvent_seq[:K]
                desc_k = desc_seq[:K]
                x = self._step_features(mol_vec, sol_k, desc_k, solvent_vecs)
                return self._forward_from_x(x)
            x = self._step_features(mol_vec, solvent_seq, desc_seq, solvent_vecs)
            out_t = self._forward_from_x(x)
            rf_k = torch.zeros(K, device=out_t.device, dtype=out_t.dtype)
            rf_k[:T] = out_t
            return rf_k

        x = self._step_features(mol_vec, solvent_seq, desc_seq, solvent_vecs)
        return self._forward_from_x(x)

    def forward_with_branches(self, mol_vec, solvent_seq, desc_seq, solvent_vecs):
        """返回 (rf_ens, rf_fwd, rf_bwd) 供辅助损失等使用。"""
        if self.num_channels and self.num_channels > 0:
            K = self.num_channels
            T = solvent_seq.size(0)

            def _branches_from_x(x_):
                h_ = self.step_proj(x_)
                h_ = self.pos_enc(h_)
                hf = self._encode_steps(h_)
                rff = self.readout_fwd(hf).squeeze(-1).clamp(1e-4, 1.0 - 1e-4)
                hb = self._encode_steps(h_.flip(0)).flip(0)
                rfb = self.readout_bwd(hb).squeeze(-1).clamp(1e-4, 1.0 - 1e-4)
                return 0.5 * (rff + rfb), rff, rfb

            if T >= K:
                sol_k = solvent_seq[:K]
                desc_k = desc_seq[:K]
                x = self._step_features(mol_vec, sol_k, desc_k, solvent_vecs)
                return _branches_from_x(x)
            x = self._step_features(mol_vec, solvent_seq, desc_seq, solvent_vecs)
            ens_t, ff_t, fb_t = _branches_from_x(x)
            rf_ens = torch.zeros(K, device=ens_t.device, dtype=ens_t.dtype)
            rf_fwd = torch.zeros(K, device=ff_t.device, dtype=ff_t.dtype)
            rf_bwd = torch.zeros(K, device=fb_t.device, dtype=fb_t.dtype)
            rf_ens[:T] = ens_t
            rf_fwd[:T] = ff_t
            rf_bwd[:T] = fb_t
            return rf_ens, rf_fwd, rf_bwd

        x = self._step_features(mol_vec, solvent_seq, desc_seq, solvent_vecs)
        h = self.step_proj(x)
        h = self.pos_enc(h)
        h_fwd = self._encode_steps(h)
        rf_fwd = self.readout_fwd(h_fwd).squeeze(-1).clamp(1e-4, 1.0 - 1e-4)
        h_bwd_in = h.flip(0)
        h_bwd = self._encode_steps(h_bwd_in).flip(0)
        rf_bwd = self.readout_bwd(h_bwd).squeeze(-1).clamp(1e-4, 1.0 - 1e-4)
        rf_ens = 0.5 * (rf_fwd + rf_bwd)
        return rf_ens, rf_fwd, rf_bwd
