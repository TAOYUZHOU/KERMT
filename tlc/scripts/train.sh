#!/usr/bin/env bash
set -euo pipefail

KERMT_DIR="/root/autodl-tmp/taoyuzhou/KERMT"
ENV_PREFIX="/root/autodl-tmp/taoyuzhou/conda_envs/kermt"
KERMT_PYTHON="${ENV_PREFIX}/bin/python"
TLC_DIR="${KERMT_DIR}/tlc"

export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${KERMT_DIR}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

EPOCHS="${1:-10}"
SAVE_DIR="${TLC_DIR}/results/run_ep${EPOCHS}"

echo "============================================"
echo "  KERMT TLC Rf Training"
echo "  Epochs: ${EPOCHS}"
echo "  Save:   ${SAVE_DIR}"
echo "============================================"

cd "${KERMT_DIR}"

if [ ! -f "${TLC_DIR}/data/train.csv" ]; then
  echo "Converting data..."
  $KERMT_PYTHON "${TLC_DIR}/scripts/convert_data.py"
fi

mkdir -p "${SAVE_DIR}"

$KERMT_PYTHON main.py finetune \
  --data_path "${TLC_DIR}/data/train.csv" \
  --separate_val_path "${TLC_DIR}/data/valid.csv" \
  --separate_test_path "${TLC_DIR}/data/test.csv" \
  --checkpoint_path pretrained_models/grover_base.pt \
  --save_dir "${SAVE_DIR}" \
  --dataset_type regression \
  --split_type scaffold_balanced \
  --ensemble_size 1 \
  --num_folds 1 \
  --no_features_scaling \
  --ffn_hidden_size 700 \
  --ffn_num_layers 3 \
  --bond_drop_rate 0.1 \
  --epochs "${EPOCHS}" \
  --metric mae \
  --self_attention \
  --dist_coff 0.15 \
  --max_lr 1e-4 \
  --final_lr 2e-5 \
  --dropout 0.0 \
  --batch_size 32 \
  --no_cuda

echo ""
echo "Training complete. Model saved to: ${SAVE_DIR}"
