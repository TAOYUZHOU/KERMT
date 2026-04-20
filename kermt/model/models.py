# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License

# Copyright (c) 2021 Tencent AI Lab.  All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
The KERMT models for pretraining, finetuning and fingerprint generating.
"""
from argparse import Namespace
from typing import List, Dict, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.cuda import nvtx

from kermt.data import get_atom_fdim, get_bond_fdim
from kermt.model.layers import Readout, GTransEncoder
from kermt.util.nn_utils import get_activation_function


SOLVENT_SMILES = ["CCCCCC", "CCOC(C)=O", "ClCCl", "CO", "CCOCC"]  # H, EA, DCM, MeOH, Et2O


def _parse_beta_raw(raw: torch.Tensor, num_tasks: int):
    """Map FFN raw outputs [B, 2 * num_tasks] to Beta(μ, φ) parameters per task."""
    b = raw.shape[0]
    r = raw.view(b, num_tasks, 2)
    mu = torch.sigmoid(r[..., 0])
    phi = F.softplus(r[..., 1]) + 2.0
    return mu, phi


def _beta_nll_neglogprob(mu: torch.Tensor, phi: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """Negative log-likelihood of Beta(μ, φ) at `target` (same shape as mu)."""
    t = target.clamp(eps, 1.0 - eps)
    alpha = mu * phi
    beta = (1.0 - mu) * phi
    log_prob = (
        torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + (alpha - 1.0) * torch.log(t)
        + (beta - 1.0) * torch.log(1.0 - t)
    )
    return -log_prob


class _ResBlock(nn.Module):
    """Residual block with pre-LayerNorm."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.norm(x + self.drop(self.fc2(self.act(self.fc1(x)))))


class SolventProjection(nn.Module):
    """Project 5 ratio-scaled GNN solvent embeddings to mol_dim.

    Each solvent SMILES is encoded by the shared KERMT encoder.
    The resulting vectors are scaled by their ratios, concatenated [5*mol_dim],
    then projected to [mol_dim] through a 3-layer residual MLP.
    """

    def __init__(self, mol_dim, n_solvents=5, n_blocks=3, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Linear(n_solvents * mol_dim, mol_dim)
        self.blocks = nn.ModuleList([
            _ResBlock(mol_dim, dropout) for _ in range(n_blocks)
        ])

    def forward(self, solvent_vecs, ratios):
        """
        solvent_vecs: [n_solvents, mol_dim]  (GNN-encoded, shared encoder)
        ratios:       [B, n_solvents]
        returns:      [B, mol_dim]
        """
        scaled = ratios.unsqueeze(-1) * solvent_vecs.unsqueeze(0)  # [B, 5, mol_dim]
        x = scaled.reshape(scaled.size(0), -1)                     # [B, 5*mol_dim]
        x = torch.nn.functional.gelu(self.proj_in(x))              # [B, mol_dim]
        for block in self.blocks:
            x = block(x)
        return x


class KERMTEmbedding(nn.Module):
    """
    The KERMT Embedding class. It contains the GTransEncoder.
    This GTransEncoder can be replaced by any validate encoders.
    """

    def __init__(self, args: Namespace):
        """
        Initialize the KERMTEmbedding class.
        :param args:
        """
        super(KERMTEmbedding, self).__init__()
        self.embedding_output_type = args.embedding_output_type
        edge_dim = get_bond_fdim() + get_atom_fdim()
        node_dim = get_atom_fdim()
        if not hasattr(args, "backbone"):
            print("No backbone specified in args, use gtrans backbone.")
            args.backbone = "gtrans"
        if args.backbone == "gtrans" or args.backbone == "dualtrans":
            # dualtrans is the old name.
            self.encoders = GTransEncoder(args,
                                          hidden_size=args.hidden_size,
                                          edge_fdim=edge_dim,
                                          node_fdim=node_dim,
                                          dropout=args.dropout,
                                          activation=args.activation,
                                          num_mt_block=args.num_mt_block,
                                          num_attn_head=args.num_attn_head,
                                          atom_emb_output=self.embedding_output_type,
                                          bias=args.bias,
                                          cuda=args.cuda)

    def forward(self, graph_batch: List) -> Dict:
        """
        The forward function takes graph_batch as input and output a dict. The content of the dict is decided by
        self.embedding_output_type.

        :param graph_batch: the input graph batch generated by MolCollator.
        :return: a dict containing the embedding results.
        """
        output = self.encoders(graph_batch)
        if self.embedding_output_type == 'atom':
            return {"atom_from_atom": output[0], "atom_from_bond": output[1],
                    "bond_from_atom": None, "bond_from_bond": None}  # atom_from_atom, atom_from_bond
        elif self.embedding_output_type == 'bond':
            return {"atom_from_atom": None, "atom_from_bond": None,
                    "bond_from_atom": output[0], "bond_from_bond": output[1]}  # bond_from_atom, bond_from_bond
        elif self.embedding_output_type == "both":
            return {"atom_from_atom": output[0][0], "bond_from_atom": output[0][1],
                    "atom_from_bond": output[1][0], "bond_from_bond": output[1][1]}


class AtomVocabPrediction(nn.Module):
    """
    The atom-wise vocabulary prediction task. The atom vocabulary is constructed by the context.
    """
    def __init__(self, args, vocab_size, hidden_size=None):
        """
        :param args: the argument.
        :param vocab_size: the size of atom vocabulary.
        """
        super(AtomVocabPrediction, self).__init__()
        if not hidden_size:
            hidden_size = args.hidden_size
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, embeddings):
        """
        If embeddings is None: do not go through forward pass.
        :param embeddings: the atom embeddings, num_atom X fea_dim.
        :return: the prediction for each atom, num_atom X vocab_size.
        """
        if embeddings is None:
            return None
        return self.logsoftmax(self.linear(embeddings))


class BondVocabPrediction(nn.Module):
    """
    The bond-wise vocabulary prediction task. The bond vocabulary is constructed by the context.
    """
    def __init__(self, args, vocab_size, hidden_size=None):
        """
        Might need to use different architecture for bond vocab prediction.
        :param args:
        :param vocab_size: size of bond vocab.
        :param hidden_size: hidden size
        """
        super(BondVocabPrediction, self).__init__()
        if not hidden_size:
            hidden_size = args.hidden_size
        self.linear = nn.Linear(hidden_size, vocab_size)

        # ad-hoc here
        # If TWO_FC_4_BOND_VOCAB, we will use two distinct fc layer to deal with the bond and rev bond.
        self.TWO_FC_4_BOND_VOCAB = True
        if self.TWO_FC_4_BOND_VOCAB:
            self.linear_rev = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, embeddings):
        """
        If embeddings is None: do not go through forward pass.
        :param embeddings: the atom embeddings, num_bond X fea_dim.
        :return: the prediction for each atom, num_bond X vocab_size.
        """
        if embeddings is None:
            return None
        nm_bonds = embeddings.shape[0]  # must be an odd number
        # The bond and rev bond have odd and even ids respectively. See definition in molgraph.
        ids1 = [0] + list(range(1, nm_bonds, 2))
        ids2 = list(range(0, nm_bonds, 2))
        if self.TWO_FC_4_BOND_VOCAB:
            logits = self.linear(embeddings[ids1]) + self.linear_rev(embeddings[ids2])
        else:
            logits = self.linear(embeddings[ids1] + embeddings[ids2])

        return self.logsoftmax(logits)


class FunctionalGroupPrediction(nn.Module):
    """
    The functional group (semantic motifs) prediction task. This is a graph-level task.
    """
    def __init__(self, args, fg_size):
        """
        :param args: The arguments.
        :param fg_size: The size of semantic motifs.
        """
        super(FunctionalGroupPrediction, self).__init__()
        first_linear_dim = args.hidden_size
        hidden_size = args.hidden_size

        # In order to retain maximal information in the encoder, we use a simple readout function here.
        self.readout = Readout(rtype="mean", hidden_size=hidden_size)
        # We have four branches here. But the input with less than four branch is OK.
        # Since we use BCEWithLogitsLoss as the loss function, we only need to output logits here.
        self.linear_atom_from_atom = nn.Linear(first_linear_dim, fg_size)
        self.linear_atom_from_bond = nn.Linear(first_linear_dim, fg_size)
        self.linear_bond_from_atom = nn.Linear(first_linear_dim, fg_size)
        self.linear_bond_from_bond = nn.Linear(first_linear_dim, fg_size)

    def forward(self, embeddings: Dict, ascope: List, bscope: List) -> Dict:
        """
        The forward function of semantic motif prediction. It takes the node/bond embeddings, and the corresponding
        atom/bond scope as input and produce the prediction logits for different branches.
        :param embeddings: The input embeddings are organized as dict. The output of KERMTEmbedding.
        :param ascope: The scope for bonds. Please refer BatchMolGraph for more details.
        :param bscope: The scope for aotms. Please refer BatchMolGraph for more details.
        :return: a dict contains the predicted logits.
        """

        preds_atom_from_atom, preds_atom_from_bond, preds_bond_from_atom, preds_bond_from_bond = \
            None, None, None, None

        if embeddings["bond_from_atom"] is not None:
            preds_bond_from_atom = self.linear_bond_from_atom(self.readout(embeddings["bond_from_atom"], bscope))
        if embeddings["bond_from_bond"] is not None:
            preds_bond_from_bond = self.linear_bond_from_bond(self.readout(embeddings["bond_from_bond"], bscope))

        if embeddings["atom_from_atom"] is not None:
            preds_atom_from_atom = self.linear_atom_from_atom(self.readout(embeddings["atom_from_atom"], ascope))
        if embeddings["atom_from_bond"] is not None:
            preds_atom_from_bond = self.linear_atom_from_bond(self.readout(embeddings["atom_from_bond"], ascope))

        return {"atom_from_atom": preds_atom_from_atom, "atom_from_bond": preds_atom_from_bond,
                "bond_from_atom": preds_bond_from_atom, "bond_from_bond": preds_bond_from_bond}


class KermtTask(nn.Module):
    """
    The pretrain module.
    """
    def __init__(self, args, kermt, atom_vocab_size, bond_vocab_size, fg_size):
        super(KermtTask, self).__init__()
        self.kermt = kermt
        self.av_task_atom = AtomVocabPrediction(args, atom_vocab_size)
        self.av_task_bond = AtomVocabPrediction(args, atom_vocab_size)
        self.bv_task_atom = BondVocabPrediction(args, bond_vocab_size)
        self.bv_task_bond = BondVocabPrediction(args, bond_vocab_size)

        self.fg_task_all = FunctionalGroupPrediction(args, fg_size)

        self.embedding_output_type = args.embedding_output_type

    @staticmethod
    def get_loss_func(args: Namespace) -> Callable:
        """
        The loss function generator.
        :param args: the arguments.
        :return: the loss function for KermtTask.
        """
        def loss_func(preds, targets, dist_coff=args.dist_coff):
            """
            The loss function for KermtTask.
            :param preds: the predictions.
            :param targets: the targets.
            :param dist_coff: the default disagreement coefficient for the distances between different branches.
            :return:
            """
            av_task_loss = nn.NLLLoss(ignore_index=0, reduction="mean")  # same for av and bv

            fg_task_loss = nn.BCEWithLogitsLoss(reduction="mean")
            # av_task_dist_loss = nn.KLDivLoss(reduction="mean")
            av_task_dist_loss = nn.MSELoss(reduction="mean")
            fg_task_dist_loss = nn.MSELoss(reduction="mean")
            sigmoid = nn.Sigmoid()

            av_atom_loss, av_bond_loss, av_dist_loss = 0.0, 0.0, 0.0
            fg_atom_from_atom_loss, fg_atom_from_bond_loss, fg_atom_dist_loss = 0.0, 0.0, 0.0
            bv_atom_loss, bv_bond_loss, bv_dist_loss = 0.0, 0.0, 0.0
            fg_bond_from_atom_loss, fg_bond_from_bond_loss, fg_bond_dist_loss = 0.0, 0.0, 0.0

            if preds["av_task"][0] is not None:
                av_atom_loss = av_task_loss(preds['av_task'][0], targets["av_task"])
                fg_atom_from_atom_loss = fg_task_loss(preds["fg_task"]["atom_from_atom"], targets["fg_task"])

            if preds["av_task"][1] is not None:
                av_bond_loss = av_task_loss(preds['av_task'][1], targets["av_task"])
                fg_atom_from_bond_loss = fg_task_loss(preds["fg_task"]["atom_from_bond"], targets["fg_task"])

            if preds["bv_task"][0] is not None:
                bv_atom_loss = av_task_loss(preds['bv_task'][0], targets["bv_task"])
                fg_bond_from_atom_loss = fg_task_loss(preds["fg_task"]["bond_from_atom"], targets["fg_task"])

            if preds["bv_task"][1] is not None:
                bv_bond_loss = av_task_loss(preds['bv_task'][1], targets["bv_task"])
                fg_bond_from_bond_loss = fg_task_loss(preds["fg_task"]["bond_from_bond"], targets["fg_task"])

            if preds["av_task"][0] is not None and preds["av_task"][1] is not None:
                av_dist_loss = av_task_dist_loss(preds['av_task'][0], preds['av_task'][1])
                fg_atom_dist_loss = fg_task_dist_loss(sigmoid(preds["fg_task"]["atom_from_atom"]),
                                                      sigmoid(preds["fg_task"]["atom_from_bond"]))

            if preds["bv_task"][0] is not None and preds["bv_task"][1] is not None:
                bv_dist_loss = av_task_dist_loss(preds['bv_task'][0], preds['bv_task'][1])
                fg_bond_dist_loss = fg_task_dist_loss(sigmoid(preds["fg_task"]["bond_from_atom"]),
                                                      sigmoid(preds["fg_task"]["bond_from_bond"]))

            av_loss = av_atom_loss + av_bond_loss
            bv_loss = bv_atom_loss + bv_bond_loss
            fg_atom_loss = fg_atom_from_atom_loss + fg_atom_from_bond_loss
            fg_bond_loss = fg_bond_from_atom_loss + fg_bond_from_bond_loss

            fg_loss = fg_atom_loss + fg_bond_loss
            fg_dist_loss = fg_atom_dist_loss + fg_bond_dist_loss

            # dist_loss = av_dist_loss + bv_dist_loss + fg_dist_loss
            # print("%.4f %.4f %.4f %.4f %.4f %.4f"%(av_atom_loss,
            #                                       av_bond_loss,
            #                                       fg_atom_loss,
            #                                       fg_bond_loss,
            #                                       av_dist_loss,
            #                                       fg_dist_loss))
            # return av_loss + fg_loss + dist_coff * dist_loss
            overall_loss = av_loss + bv_loss + fg_loss + dist_coff * av_dist_loss + \
                           dist_coff * bv_dist_loss + fg_dist_loss

            return overall_loss, av_loss, bv_loss, fg_loss, av_dist_loss, bv_dist_loss, fg_dist_loss

        return loss_func

    def forward(self, graph_batch: List):
        """
        The forward function.
        :param graph_batch:
        :return:
        """
        _, _, _, _, _, a_scope, b_scope, _ = graph_batch
        a_scope = a_scope.data.cpu().numpy().tolist()

        nvtx.range_push("embedding")
        embeddings = self.kermt(graph_batch)
        nvtx.range_pop() # embedding

        nvtx.range_push("atom_from_atom")
        av_task_pred_atom = self.av_task_atom(
            embeddings["atom_from_atom"])  # if None: means not go through this fowward
        nvtx.range_pop() # atom_from_atom
        nvtx.range_push("atom_from_bond") 
        av_task_pred_bond = self.av_task_bond(embeddings["atom_from_bond"])
        nvtx.range_pop() # atom_from_bond

        nvtx.range_push("bond_from_atom")
        bv_task_pred_atom = self.bv_task_atom(embeddings["bond_from_atom"])
        nvtx.range_pop() # bond_from_atom

        nvtx.range_push("bond_from_bond")
        bv_task_pred_bond = self.bv_task_bond(embeddings["bond_from_bond"])
        nvtx.range_pop() # bond_from_bond

        nvtx.range_push("fg_task_all")
        fg_task_pred_all = self.fg_task_all(embeddings, a_scope, b_scope)
        nvtx.range_pop() # fg_task_all

        return {"av_task": (av_task_pred_atom, av_task_pred_bond),
                "bv_task": (bv_task_pred_atom, bv_task_pred_bond),
                "fg_task": fg_task_pred_all}


class KermtFpGeneration(nn.Module):
    """
    KermtFpGeneration class.
    It loads the pre-trained model and produce the fingerprints for input molecules.
    """
    def __init__(self, args):
        """
        Init function.
        :param args: the arguments.
        """
        super(KermtFpGeneration, self).__init__()

        self.fingerprint_source = args.fingerprint_source
        self.iscuda = args.cuda

        self.kermt = KERMTEmbedding(args)
        self.readout = Readout(rtype="mean", hidden_size=args.hidden_size)

    def forward(self, batch, features_batch):
        """
        The forward function.
        It takes graph batch and molecular feature batch as input and produce the fingerprints of this molecules.
        :param batch:
        :param features_batch:
        :return:
        """
        _, _, _, _, _, a_scope, b_scope, _ = batch

        output = self.kermt(batch)
        # Share readout
        mol_atom_from_bond_output = self.readout(output["atom_from_bond"], a_scope)
        mol_atom_from_atom_output = self.readout(output["atom_from_atom"], a_scope)

        if self.fingerprint_source == "bond" or self.fingerprint_source == "both":
            mol_bond_from_atom_output = self.readout(output["bond_from_atom"], b_scope)
            mol_bond_from_bodd_output = self.readout(output["bond_from_bond"], b_scope)

        if features_batch[0] is not None:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            if self.iscuda:
                features_batch = features_batch.cuda()
            features_batch = features_batch.to(output["atom_from_atom"])
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
        else:
            features_batch = None

        if self.fingerprint_source == "atom":
            fp = torch.cat([mol_atom_from_atom_output, mol_atom_from_bond_output], 1)
        elif self.fingerprint_source == "bond":
            fp = torch.cat([mol_bond_from_atom_output, mol_bond_from_bodd_output], 1)
        else:
            # the both case.
            fp = torch.cat([mol_atom_from_atom_output, mol_atom_from_bond_output,
                            mol_bond_from_atom_output, mol_bond_from_bodd_output], 1)
        if features_batch is not None:
            fp = torch.cat([fp, features_batch], 1)
        return fp


class KermtFinetuneTask(nn.Module):
    """
    The finetune
    """
    def __init__(self, args):
        super(KermtFinetuneTask, self).__init__()

        self.hidden_size = args.hidden_size
        self.iscuda = args.cuda

        self.kermt = KERMTEmbedding(args)

        if args.self_attention:
            self.readout = Readout(rtype="self_attention", hidden_size=self.hidden_size,
                                   attn_hidden=args.attn_hidden,
                                   attn_out=args.attn_out)
        else:
            self.readout = Readout(rtype="mean", hidden_size=self.hidden_size)

        self.use_solvent_enc = getattr(args, 'solvent_emb_dim', 0) > 0
        self.n_solvent_dims = getattr(args, 'n_solvent_dims', 5)
        self.n_cross_dims = getattr(args, 'n_cross_dims', 10)
        if self.use_solvent_enc:
            if args.self_attention:
                mol_dim = args.hidden_size * args.attn_out
            else:
                mol_dim = args.hidden_size
            self._mol_dim = mol_dim
            self._build_solvent_graph(args)
            self.solvent_proj = SolventProjection(mol_dim, self.n_solvent_dims)
            raw_npz_w = getattr(args, "_tlc_raw_npz_features_size", None)
            if raw_npz_w is None:
                raw_npz_w = args.features_size
                setattr(args, "_tlc_raw_npz_features_size", int(raw_npz_w))
            n_desc_dims = raw_npz_w - self.n_solvent_dims - self.n_cross_dims
            args.features_size = mol_dim + n_desc_dims

        self.use_beta_regression = (
            args.dataset_type == "regression"
            and getattr(args, "regression_loss", "mse") == "beta_nll"
        )
        _saved_out = args.output_size
        if self.use_beta_regression:
            args.output_size = _saved_out * 2
        self.mol_atom_from_atom_ffn = self.create_ffn(args)
        self.mol_atom_from_bond_ffn = self.create_ffn(args)
        args.output_size = _saved_out
        #self.ffn = nn.ModuleList()
        #self.ffn.append(self.mol_atom_from_atom_ffn)
        #self.ffn.append(self.mol_atom_from_bond_ffn)

        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        # Note: args.features_dim is set according the real loaded features data
        if args.features_only:
            first_linear_dim = args.features_size + args.features_dim
        else:
            if args.self_attention:
                first_linear_dim = args.hidden_size * args.attn_out
                # TODO: Ad-hoc!
                # if args.use_input_features:
                # first_linear_dim += args.features_dim
                # TODO(sveccham): Verify that this is correct.
                first_linear_dim += args.features_size 
            else:
                # TODO(sveccham): Verify that this is correct.
                first_linear_dim = args.hidden_size + args.features_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        # TODO: ffn_hidden_size
        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        return nn.Sequential(*ffn)

    @staticmethod
    def get_loss_func(args):
        use_beta = (
            args.dataset_type == "regression"
            and getattr(args, "regression_loss", "mse") == "beta_nll"
        )
        num_tasks = getattr(args, "num_tasks", 1)

        def loss_func(preds, targets,
                      dt=args.dataset_type,
                      dist_coff=args.dist_coff):

            if dt == 'classification':
                pred_loss = nn.BCEWithLogitsLoss(reduction='none')
            elif dt == 'regression' and not use_beta:
                pred_loss = nn.MSELoss(reduction='none')
            elif dt == 'regression' and use_beta:
                pred_loss = None
            else:
                raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')

            # print(type(preds))
            # TODO: Here, should we need to involve the model status? Using len(preds) is just a hack.
            if type(preds) is not tuple:
                # in eval mode.
                if use_beta:
                    return nn.MSELoss(reduction='none')(preds, targets)
                return pred_loss(preds, targets)

            # in train mode.
            if use_beta:
                mu_a, phi_a = _parse_beta_raw(preds[0], num_tasks)
                mu_b, phi_b = _parse_beta_raw(preds[1], num_tasks)
                nll_a = _beta_nll_neglogprob(mu_a, phi_a, targets)
                nll_b = _beta_nll_neglogprob(mu_b, phi_b, targets)
                dist = (mu_a - mu_b) ** 2
                return nll_a + nll_b + dist_coff * dist

            dist_loss = nn.MSELoss(reduction='none')
            dist = dist_loss(preds[0], preds[1])
            pred_loss1 = pred_loss(preds[0], targets)
            pred_loss2 = pred_loss(preds[1], targets)
            return pred_loss1 + pred_loss2 + dist_coff * dist

        return loss_func

    def _build_solvent_graph(self, args):
        """Pre-build molecular graphs for 5 solvents (stored on CPU)."""
        from kermt.data.molgraph import mol2graph
        bg = mol2graph(SOLVENT_SMILES, {}, args)
        self._solvent_graph_cpu = bg.get_components()

    def _encode_solvents(self):
        """Encode 5 solvent SMILES through the shared KERMT encoder → [5, mol_dim]."""
        device = next(self.parameters()).device
        graph = tuple(
            c.to(device) if isinstance(c, torch.Tensor) else c
            for c in self._solvent_graph_cpu
        )
        output = self.kermt(graph)
        _, _, _, _, _, a_scope, _, _ = graph
        v_atom = self.readout(output["atom_from_atom"], a_scope)
        v_bond = self.readout(output["atom_from_bond"], a_scope)
        if isinstance(v_atom, tuple):
            v_atom = v_atom[0].flatten(start_dim=1)
            v_bond = v_bond[0].flatten(start_dim=1)
        return (v_atom + v_bond) / 2.0

    def _process_features(self, features_batch, ref_tensor):
        """Convert raw numpy features to tensor; if GNN solvent encoding is on,
        replace raw solvent ratios with projected GNN embeddings."""
        if features_batch[0] is None:
            return None
        features_batch = torch.from_numpy(np.stack(features_batch)).float()
        if self.iscuda:
            features_batch = features_batch.cuda()
        features_batch = features_batch.to(ref_tensor)
        if len(features_batch.shape) == 1:
            features_batch = features_batch.view([1, features_batch.shape[0]])

        if self.use_solvent_enc:
            ratios = features_batch[:, :self.n_solvent_dims]
            desc = features_batch[:, self.n_solvent_dims + self.n_cross_dims:]
            solvent_vecs = self._encode_solvents()                # [5, mol_dim]
            solv_proj = self.solvent_proj(solvent_vecs, ratios)   # [B, mol_dim]
            features_batch = torch.cat([solv_proj, desc], dim=1)  # [B, mol_dim+6]

        return features_batch

    def forward(self, batch, features_batch):
        _, _, _, _, _, a_scope, _, _ = batch

        output = self.kermt(batch)
        mol_atom_from_bond_output = self.readout(output["atom_from_bond"], a_scope)
        mol_atom_from_atom_output = self.readout(output["atom_from_atom"], a_scope)

        features_batch = self._process_features(features_batch, output["atom_from_atom"])

        if features_batch is not None:
            mol_atom_from_atom_output = torch.cat([mol_atom_from_atom_output, features_batch], 1)
            mol_atom_from_bond_output = torch.cat([mol_atom_from_bond_output, features_batch], 1)

        if self.training:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            return atom_ffn_output, bond_ffn_output
        else:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            if self.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output)
                bond_ffn_output = self.sigmoid(bond_ffn_output)
                output = (atom_ffn_output + bond_ffn_output) / 2
            elif self.use_beta_regression:
                nt = atom_ffn_output.shape[1] // 2
                mu_a, _ = _parse_beta_raw(atom_ffn_output, nt)
                mu_b, _ = _parse_beta_raw(bond_ffn_output, nt)
                output = (mu_a + mu_b) / 2.0
            else:
                output = (atom_ffn_output + bond_ffn_output) / 2

        return output
