# KERMT TLC 训练实验对比与说明

本文档汇总 **方案 A（Plan A / `KermtFinetuneTask`）** 与 **方案 B（自回归）** 若干版本在 **同一套 `tlc/data` 划分** 下的测试表现对比，并记录讨论结论（为何简单 v1 续训可强于部分 v3 / Beta NLL 设定）。  
指标以各 `results/<run>/fold_0/test_result.csv` 或训练收尾日志 / `eval_and_plot_*` 汇总为准；不同脚本若样本对齐方式不同，数值可能有微小差异。

---

## 1. 什么是「表格回归」（tabular regression）

在非正式讨论里，**表格回归**指：把每个样本看成 **一行固定维度的特征向量**（如溶剂比例、交叉项、分子描述符等），目标是一个标量（此处为 Rf），用 **线性或非线性的「表格模型」**（广义上包括把向量送进 MLP / FFN）做回归。

在 **方案 A v1（`solvent_emb_dim: 0`）** 里，KERMT 图网络先产出 **分子向量**，再与 **21 维 NPZ 向量拼接**，最后经 FFN 输出 Rf——后半段「固定维向量 → 标量」与经典表格回归 **形式相近**；区别是分子侧特征来自 GNN 而非手工表列。称「接近表格回归」强调的是 **溶剂与描述符以显式数值进入头**，优化路径短、与 MSE+标准化常见搭配一致。

---

## 2. 主要实验结果对比（Test，约 4737 样本量级）

| 实验目录 / 说明 | 方案 | 结构要点 | Test MAE（约） | 备注 |
|-----------------|------|----------|----------------|------|
| `with_features_v1_dirty_finetune5ep` | A | 无 GNN 溶剂分支；NPZ 直接拼接；**MSE + 目标 StandardScaler**；从 **v1_dirty** 热启仅 **5 epoch** | **~0.036** | 强基线：表征与头已在同任务上收敛 |
| `with_features_v1_cleaned_finetune5ep_beta_nll` | A | 同上结构；**Beta NLL**；无目标标准化；从 v1_dirty 热启 | **~0.042** | 损失与评价刻度与纯 MSE 线不完全一致 |
| `with_features_v3_solv_gnn_cleaned_beta_nll` | A | **共享 KERMT** 编码 5 溶剂 + `SolventProjection` 替代 NPZ 中比例块；Beta NLL；多从 grover 训 | **~0.14** | 难优化 / 与显式比例信息重叠等（见第 3 节） |
| `with_features_v3_solv_gnn_beta_nll_ep300_es100_bs128` | A | 同上；300 epoch、早停 100、batch 128 | **~0.14** | 长训后仍明显弱于 v1 续训 |
| `autoregressive_v3_solv_gnn_ep300_es100_bs128` | B | 序列自回归 + Beta；早停于 ~184 epoch，best val ~epoch 84 | **~0.099** | 与「单行单 Rf」的 A 任务不同，不宜仅比结构复杂度 |
| `autoregressive_v3_solv_gnn`（及同类 v3 自回归） | B | 配置见各自 `effective_config.yaml` | **~0.10 量级** | 序列逐步误差与 v1 标量回归不对等 |

**读数来源示例**：`with_features_v1_dirty_finetune5ep/quiet.log` 中 `Model 0 test mae = 0.036331`；v3 Beta ep300 与自回归 ep300 来自对应 `nohup_train.log` / `plot_run.log` 汇总。

---

## 3. 讨论摘要：为何「更复杂」不一定更好？

### 3.1 训练起点与有效训练预算

- **v1 dirty finetune5ep** 是在 **已在同一数据形态上训好的 checkpoint** 上极短微调，编码器与 FFN 已对齐任务。
- 多组 **v3 Plan A** 从 **grover_base** 等起训，等价于 **整条链路重学**；参数量与融合方式更复杂，需要 **更多轮次与调参** 才可能达到 v1 水平。

### 3.2 溶剂：共享预训练 ≠ 没有「新路径」

- **v3** 中 5 个溶剂 SMILES 走 **与待测分子相同的 `KERMTEmbedding`（共享权重）**，再经 **SolventProjection + 比例** 融合，**替换** NPZ 里原本的溶剂比例（及绑定结构）再与描述符拼接。
- 因此不是「完全另一套不共享的溶剂 GNN」，而是：**在共享编码器之上，用可学习融合替代了表格里已有的显式比例信息**；若该融合未训好，可能 **不如** 直接把比例拼进 FFN（v1）。

### 3.3 「与 NPZ 冗余」指什么？

指 **信息内容重叠**（比例、体系信息已在 NPZ 中），而非「两个独立溶剂模型」。非线性重参数化在数据量与优化有限时，不一定优于简单的显式特征 + MSE。

### 3.4 Beta NLL 与「很快就有不错的 MAE」

- **Beta NLL 本身不保证比 MSE 收敛更快**。
- 若前几轮验证 MAE 已较低，通常主因是：**强热启或强表征** + **预测用 μ（如 sigmoid 初值在 0.5 附近对 [0,1] Rf 合理）** + **已过至少一个 epoch 的头更新**，而不是损失形式必然更优。

### 3.5 方案 B 与方案 A 的指标不可硬比「结构优劣」

自回归在 **序列步** 上预测，与 v1 **一行一个 Rf** 任务定义不同；Test MAE 差一截 **不能** 简单归结为「B 比 A 差」，而是任务难度与评价对象不同。

---

## 4. 绘图与复现命令（节选）

- 方案 A：`tlc/scripts/eval_and_plot_plan_a.py --result_dir tlc/results/<run>`
- 方案 B：`tlc/scripts/eval_and_plot_autoregressive.py --model_dir tlc/results/<run>`

详见 `tlc/README.md`。
---

*文档生成/更新说明：与对话中关于 v1 / v3 / Beta NLL / 自回归的讨论对齐；具体数值请以各次训练目录内日志与 `test_result.csv` 为准。*
