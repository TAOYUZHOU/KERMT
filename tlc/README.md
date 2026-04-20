# KERMT × TLC Rf 预测

使用 KERMT (GROVER base) 预训练模型对 TLC Rf 值进行回归预测。

## 目录结构

```
tlc/
├── README.md
├── configs/
│   ├── tlc_with_features.yaml     # 方案 A 配置
│   ├── tlc_autoregressive.yaml    # 方案 B 配置
│   └── (规划中) tlc_bidirectional_c_v1.yaml  # 方案 C v1
├── data/
│   ├── train.csv / valid.csv / test.csv                # smiles, Rf
│   ├── train_features.npz / valid_features.npz / ...   # 溶剂+交叉+描述符 [N,21]
│   └── train_sequences.json / valid_sequences.json / ...# 方案 B 序列数据
├── scripts/
│   ├── convert_data.py            # UniMol 格式 → KERMT CSV
│   ├── build_features.py          # 生成 21 维特征 npz
│   ├── build_sequences.py         # 按分子分组 + 溶剂体系排序 → JSON
│   ├── train.py                   # 方案 A 训练入口 (config-driven)
│   ├── predict.py                 # 方案 A 预测 + 评估
│   ├── train_autoregressive.py    # 方案 B 训练入口
│   ├── autoregressive_model.py    # 方案 B 模型定义 (新: GNN 溶剂投影 / 旧: SolventMLP)
│   ├── (规划中) train_bidirectional_c_v1.py / bidirectional_c_model.py  # 方案 C v1
│   ├── eval_and_plot_plan_a.py    # 方案 A：重跑 predict + Parity 图（与 B 同版式）
│   ├── eval_and_plot_autoregressive.py  # 方案 B/C：--plan autoregressive|bidirectional_c，plots 产物布局与 v3 自回归一致
│   ├── tlc_plot_common.py         # A/B 共用绘图
│   ├── config_loader.py           # YAML → argparse
│   └── plot_results.py            # 仅辅助：训练 log 曲线（勿与 predictions 错行对齐）
└── results/
    ├── with_features/             # 方案 A 输出
    ├── autoregressive/            # 方案 B 输出
    └── (规划中) bidirectional_c_v1_*/  # 方案 C v1
```

## 数据来源

脚本 `convert_data.py` / `build_features.py` / `build_sequences.py` 通过环境变量 **`SRC_DATA_DIR`** 指向上游目录；**默认**为  
`unimol_tlc/merged_data_molecule_split_v2_rf_stratified_heavy70_descriptor/`。  
若需使用本仓库内的 **增强划分 + `cleaned`** 子集，请显式设置，例如：

`SRC_DATA_DIR=tlc/merged_data_molecule_split_v2_rf_stratified_heavy70_descriptor_aug_extreme/cleaned`

（当前磁盘上 `tlc/data/` 行数与该 `cleaned` 下 `train_data.csv` 等 **一致**，表明近期生成数据时使用了该源而非脚本默认值。）

原始 UniMol 格式列说明：

| 列 | 说明 |
|---|---|
| COMPOUND_SMILES | 分子 SMILES |
| H, EA, DCM, MeOH, Et2O | 溶剂组成比例 |
| Rf | 目标值 (0~1) |
| MW, TPSA, LogP, HBD, HBA, NROTBs | 分子描述符 |

## 数据准备

```bash
cd /root/autodl-tmp/taoyuzhou/KERMT

# 1) 转换 CSV (smiles, Rf)
python tlc/scripts/convert_data.py

# 2) 生成 21 维特征 npz (溶剂5 + 交叉10 + 描述符6)
python tlc/scripts/build_features.py

# 3) 生成方案 B 序列数据 (按分子分组, 溶剂体系内极性升序)
python tlc/scripts/build_sequences.py
```

## 方案 A：回归 + 溶剂特征

KERMT GNN 编码分子 → 拼接 21 维特征 → FFN → Rf

```bash
# 训练 (GPU 0, 10 epochs)
CUDA_VISIBLE_DEVICES=0 python -u tlc/scripts/train.py \
    --config tlc/configs/tlc_with_features.yaml \
    --epochs 10

# 覆盖任意超参
CUDA_VISIBLE_DEVICES=0 python -u tlc/scripts/train.py \
    --config tlc/configs/tlc_with_features.yaml \
    --epochs 50 --batch_size 64 --max_lr 3e-4

# 预测 + 评估 (自动检测 config)
CUDA_VISIBLE_DEVICES=0 python tlc/scripts/predict.py \
    --checkpoint_dir tlc/results/with_features/fold_0/model_0
```

**正式评测与 Parity 图（推荐）**：对 train / valid / test 分别调用 `predict.py` 重算预测，再画单 split 与 `parity_combined_*.png`（与方案 B 同一套绘图逻辑）。

```bash
cd /root/autodl-tmp/taoyuzhou/KERMT

CUDA_VISIBLE_DEVICES=0 python tlc/scripts/eval_and_plot_plan_a.py \
    --result_dir tlc/results/with_features_v1_dirty

# 可选：指定训练 log（用于 loss 曲线；默认自动尝试 results 旁与 tlc/results 下）
# python tlc/scripts/eval_and_plot_plan_a.py --result_dir ... --train_log tlc/results/with_features_train.log

# 仅快速画测试集、不加载 GPU（只读 fold_0/test_result.csv，与「整表 predict」样本数可能不同）
# python tlc/scripts/eval_and_plot_plan_a.py --result_dir ... --artifact_only
```

产出目录：`<result_dir>/plots/`（含 `train_predictions.csv`、`valid_predictions.csv`、`test_predictions.csv`、`parity_Train|Valid|Test_*.png`、`parity_combined_*.png`、`training_curves_*.png`）。

## 方案 B：自回归单调 Rf 预测

KERMT GNN → mol_vec → 溶剂融合（新checkpoint 为 GNN 溶剂编码；旧权重为 15 维 MLP）→ Beta 回归自回归头，体系内 Rf 单调约束。

```bash
# 训练 (config-driven, 推荐)
CUDA_VISIBLE_DEVICES=1 python -u tlc/scripts/train_autoregressive.py \
    --config tlc/configs/tlc_autoregressive.yaml

# 覆盖任意超参
CUDA_VISIBLE_DEVICES=1 python -u tlc/scripts/train_autoregressive.py \
    --config tlc/configs/tlc_autoregressive.yaml \
    --epochs 50 --batch_size 8 --lr 5e-5

# 不冻结 encoder (微调全模型)
CUDA_VISIBLE_DEVICES=1 python -u tlc/scripts/train_autoregressive.py \
    --config tlc/configs/tlc_autoregressive.yaml \
    --freeze_encoder false \
    --save_dir tlc/results/autoregressive_finetune
```

**正式评测与 Parity 图（推荐）**：加载目录下 `best_model.pt`，在 train/valid/test 序列 JSON 上推理后绘图（与方案 A 同版式）。旧版 `SolventMLP` 权重会自动走兼容分支；新版 GNN 溶剂权重走当前 `KermtAutoregressive`。

```bash
cd /root/autodl-tmp/taoyuzhou/KERMT

CUDA_VISIBLE_DEVICES=0 python tlc/scripts/eval_and_plot_autoregressive.py \
    --model_dir tlc/results/autoregressive_v1_dirty

# 可选指定 GPU：--gpu 0
# 默认输出：<model_dir>/plots/
```

**方案 B 评测怎么做（与方案 C 口径对齐）**：`eval_and_plot_autoregressive.py` 对每个序列调用 `predict_sequence(..., teacher_forcing_ratio=0.0)`，得到每一步的 **预测 Rf** `rf_preds[t]`（实现上：体系/序列**首步**用 Beta 头的 `μ`；**后续步**在 `autoregressive_model.py` 里用 `rf_t = prev_rf + μ · (1 - prev_rf)`，使沿前向步序 **预测值单调不减**）。真值取 JSON 里每步的 `rf`。把所有序列、所有步的 `(y_true, y_pred)` **展平成长向量**，再算 **MAE、RMSE、R²** 并画 Parity 图——**不是**只对「首步 μ」或「未累加的残差」单独报 R²/MAE；日志里若出现逐步 Beta NLL，与评测用的逐步 **Rf** 可以并存，但 **对外报告指标以逐步 Rf 为准**。

## 方案 C v1（规划中）：双向 BERT 式序列回归

在 **`autoregressive_v3_solv_gnn`**（配置见 `tlc/configs/tlc_autoregressive_v3*.yaml`：GNN 溶剂编码 + 序列 Beta 自回归）基础上的演进方向，**尚未实现**；本节为设计备忘，便于对齐实现与实验。

### 目标与与方案 B 的差异

| 维度 | 方案 B（v3 自回归） | 方案 C v1（规划） |
|------|---------------------|-------------------|
| 序列方向 | 单向自回归（极性有序步进） | **双向**：前向 + 反向两条「BERT 类」回归路径，联合利用上下文 |
| 步间差分惩罚（规划） | 推理时 `rf_t = prev_rf + μ(1−prev_rf)`，**预测沿步序单调不减**（见上「方案 B 评测」） | **物理解释（C）**：序列与 **极性溶剂比例递增** 对齐时，希望 **Rf 单调不减**（inductive bias）。**前向**：相邻增量 `Δ = Rf_t−Rf_{t−1}`，**只罚负增量**（沿比例递增却 Rf 下降），`relu(−Δ)=relu(Rf_{t−1}−Rf_t)`。**反向**：沿 **从后往前** 的步序，一步对应「比例相对上一点变小」；**罚该步上 Rf 仍上升** 的正增量（与「比例往回走却 Rf 往上跳」相反于化学直觉）。实现见 `scheme_c_loss`；若与 v3 头冲突再改版，但单调方向与上表一致。 |
| Encoder | 单路 | **双向共享同一 KERMT 分子 encoder（及溶剂侧共享子模块）**，仅前向/反向 **头或融合方式分叉** |
| Rf 输出头 | Beta（v3） | **仍为 Beta 分布头**（与 v3 一致） |
| 分子侧描述符 | 经投影/嵌入与高维融合（随 v3 头结构） | **RDKit 6 维不做升维**，以 **原始 6 维** 与 GNN 分子向量拼接 |
| 溶剂侧融合维度 | 当前 v3：共享溶剂 GNN + 比例等 | 融合时拼接维度为 **5 × (d_emb + 6)**：5 种溶剂各一路 **溶剂 embedding + 同一套 6 维 RDKit 描述符**（描述符为**分子级**，与 v3 数据一致，按槽位重复拼接进每路溶剂向量） |

### 张量形状约定（规划）

- 记 GROVER/KERMT 分子图嵌入维度为 **`d_emb`**（与现有 `mol_dim` 一致或可配置）。
- **分子表征（单槽）**：`[mol_GNN || desc_6]`，长度为 **`d_emb + 6`**（6 维为与 `build_features` / 序列 JSON 中一致的 MW、TPSA、LogP、HBD、HBA、NROTBs，**与训练时 z-score 规则对齐**）。
- **溶剂融合输入**：对 5 个溶剂各形成 **`(d_emb + 6)`**（溶剂 GNN embedding + 同上 **6 维分子 RDKit**，保证描述符进入溶剂分支）；**沿溶剂维拼接**得到 **`5 · (d_emb + 6)`**。
- **MLP 后**：经多层 MLP 后再 **整形/映射回 `(d_emb + 6)`** 的语义空间（与正向/反向头接口一致）。
- **双向汇合**：前向、反向各输出一路 **`(d_emb + 6)`**（或经同一维度约定对齐后），**最终拼接长度为 `2 · (d_emb + 6)`**，再接入 **Beta Rf 头**（已定）。

### 训练与数据

- **数据管线**：仍基于 `build_sequences.py` 的序列 JSON；需在样本中保证每步可取得 **6 维描述符**（与现有序列字段 `desc` 对齐）。
- **损失**：主干仍为逐步 **Beta NLL**。**前向**正则：对 **`Δ = \hat Rf_t−\hat Rf_{t−1}`** 施加 **`relu(−Δ)`**，即 **只惩罚沿比例递增方向的 Rf 下降**。**反向**正则：在 **翻转序列** 上取相邻差分并 **`relu` 正部**，语义为 **惩罚沿「比例递减」行走时 Rf 仍上升**；与 `relu(−Δ)` 在同一批边上等价，用 `λ_fwd`、`λ_rev` 分别加权（可合并理解为对「单调不减」先验的双重强调）。
- **配置/脚本**：`tlc_bidirectional_c_v1.yaml`、`train_bidirectional_c_v1.py`、`bidirectional_c_model.py`；结果目录建议 `tlc/results/bidirectional_c_v1_*/`。**评测与 Parity**（与方案 B 相同目录布局）：`eval_and_plot_autoregressive.py --plan bidirectional_c --model_dir <save_dir>`，或等价包装 `eval_and_plot_bidirectional_c.py --model_dir <save_dir>`；**只需 `best_model.pt`**。可选 `--title_prefix` / `--suptitle` 区分实验名。

### 惩罚项举例说明（方案 C）

以下用**同一套逐步预测** `\hat Rf_0,\hat Rf_1,\hat Rf_2`（对应序列里 3 个溶剂点）说明**加在总损失上的额外项**（实现时再定系数 `λ_fwd`、`λ_rev`，总损失示意：`L = L_{\mathrm{BetaNLL}} + λ_{\mathrm{fwd}} L_{\mathrm{fwd}} + λ_{\mathrm{rev}} L_{\mathrm{rev}}`）。

**1）前向：极性比例递增 → 希望 Rf 不降 → 惩罚「负增量」**

- **序约定**：`t` 增大 = **极性溶剂比例沿序列递增**；希望 **`\hat Rf_t \ge \hat Rf_{t-1}`**（单调不减）。
- 定义相邻增量：`Δ_t = \hat Rf_t - \hat Rf_{t-1}`（`t=1,2,…`）。**只罚下降**：`L_{\mathrm{fwd}} = \frac{1}{T-1}\sum_t \mathrm{relu}(-Δ_t) = \frac{1}{T-1}\sum_t \max(0,\, \hat Rf_{t-1}-\hat Rf_t)`。
- **例**（`\hat Rf_0=0.50,\ \hat Rf_1=0.55,\ \hat Rf_2=0.48`，边 = 相邻**对**不是「第几个点」）：  
  - **第 1 条边**：`Δ_1=+0.05` → `relu(-Δ_1)=0`，**不罚**（上升符合先验）；  
  - **第 2 条边**：`Δ_2=-0.07` → `relu(-Δ_2)=0.07`，**罚**（比例更大一步 Rf 反而掉）。  
  故本例 `L_{\mathrm{fwd}} = \frac{1}{2}(0+0.07)=0.035`（实现为边上 **均值**；再乘 `λ_fwd`）。

**2）反向：沿「比例递减」行走 → 惩罚「正增量」**

- 将序列 **翻转** 后，从尾到头走：每一步在 **比例相对上一点变小** 的方向上看 **Rf 是否仍往上跳**。记反向步差分为 `δ`（实现上与 `bidirectional_c_model.scheme_c_loss` 中 `rv[1:]-rv[:-1]` 一致），**罚 `relu(δ)`**：只收 **正** 的那一半，对应「比例往回走却 Rf 上升」。
- **与 1）的关系**：在**同一条溶剂边上**，`relu(δ)` 与 `relu(-Δ)` 数值一致，是从 **反向行走** 表述同一单调先验；`λ_rev` 与 `λ_fwd` 可分开调权重。
- **例（示意）**：三步预测在反向衔接处若出现「从高比例点走向低比例点、Rf 却从 0.39 升到 0.46」，则该步 `δ>0` → 进入 `L_{\mathrm{rev}}`；若 Rf 随之下降则 `δ≤0`，**不罚**。

**3）和 Beta NLL 的关系**

- **Beta NLL** 仍对每一步的 Beta 分布与**该步真值 `rf`** 对齐（与方案 B 相同）。
- 上两项是**额外的形状/单调类正则**：前向 **`relu(−Δ)`** 只收 **Rf 沿比例递增下降** 的部分；反向 **`relu(δ)`** 只收 **沿比例递减行走时 Rf 反常上升** 的部分。二者共同强化 **「比例 ↑ ⇒ Rf 不减」** 的 inductive bias。

### 反向残差与评测口径（已定）

- **反向「残差」/真值差分含义**：**同一溶剂体系内、按数据序列顺序相邻两步的真值 Rf 之差**（示例：比例 **1:1** 时 Rf = **0.8**，下一步 **3:1** 时 Rf = **0.4**，则 **0.4 − 0.8 = −0.4**）。**训练惩罚与上表一致**：鼓励预测在 **比例递增** 方向 **Rf 单调不减**——**前向 `relu(−Δ)`、反向 `relu(δ)`**；真值本身可升可降，正则只是软先验，不与某条真值差分的符号强行绑定。
- **训练内部**：仍可把某一路写成「首步 Rf + 残差累加」等形式以复用自回归实现，但 **约束与监督应对齐到上述 ΔRf**。
- **评测指标（重要）**：与方案 B 现有一致，在 **每一个溶剂比例（序列每一步）上得到预测的 Rf**，将所有步的 **(y_true, y_pred)** 展平后 **统一计算 R²、MAE 等**。**不要**只对「起始 Rf + 残差链」的中间量算 R²/MAE；最终报告必须以 **逐步 Rf 预测 vs 逐步真值** 为准。

---

## 辅助：`plot_results.py`（训练曲线 / 谨慎用于散点）

仅适合拉 **训练 log** 的 loss 曲线。若画测试散点，**不要**再用「`fold_0/predictions.csv` 行」与「整份 `test.csv` 行」直接 zip：`predictions` 行数与顺序与 CSV 逐行不一致时指标会错。应用 **`--test_result_csv`** 指向 `fold_0/test_result.csv`，或改用上文 **`eval_and_plot_plan_a.py`**。

```bash
# 仅画训练曲线（可多份 log 对比）
python tlc/scripts/plot_results.py \
    --logs tlc/results/with_features_train.log tlc/results/autoregressive_v1_dirty/train.log \
    --labels "方案A" "方案B" \
    --output tlc/results/plot_aux_loss_curves.png

# 仅画与 fold_0/test_result.csv 行序一致的测试散点（单条曲线；勿与整表 test.csv zip）
python tlc/scripts/plot_results.py \
    --test_result_csv tlc/results/with_features_v1_dirty/fold_0/test_result.csv \
    --output tlc/results/plot_aux_test_result.png
```

## 当前结果 (示例：`v1_dirty` 一次评测)

| 方案 | Test MAE | Test R² | 备注 |
|------|----------|---------|------|
| A: `eval_and_plot_plan_a` 整表 test | ~0.050 | ~0.92 | `tlc/results/with_features_v1_dirty/plots/` |
| B: `eval_and_plot_autoregressive` 序列 test | ~0.129 | ~0.45 | `tlc/results/autoregressive_v1_dirty/plots/` |

## 实验对比与历史文档说明

各版本（v1 续训、v3 GNN 溶剂、Beta NLL、自回归等）的 **Test MAE 对比、讨论摘要**。
