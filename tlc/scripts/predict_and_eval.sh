#!/usr/bin/env bash
set -euo pipefail

KERMT_DIR="/root/autodl-tmp/taoyuzhou/KERMT"
ENV_PREFIX="/root/autodl-tmp/taoyuzhou/conda_envs/kermt"
KERMT_PYTHON="${ENV_PREFIX}/bin/python"
TLC_DIR="${KERMT_DIR}/tlc"

export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${KERMT_DIR}"

CHECKPOINT_DIR="${1:?Usage: $0 <checkpoint_dir> [test_csv]}"
TEST_CSV="${2:-${TLC_DIR}/data/test.csv}"
PRED_CSV="${CHECKPOINT_DIR}/predictions.csv"

echo "============================================"
echo "  KERMT TLC Rf Prediction & Evaluation"
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo "  Test data:  ${TEST_CSV}"
echo "============================================"

cd "${KERMT_DIR}"

$KERMT_PYTHON main.py predict \
  --data_path "${TEST_CSV}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --no_features_scaling \
  --output "${PRED_CSV}" \
  --no_cuda

echo ""
echo "Predictions saved to: ${PRED_CSV}"
echo ""

$KERMT_PYTHON - "${PRED_CSV}" "${TEST_CSV}" << 'PYEOF'
import csv, math, sys

pred_file, true_file = sys.argv[1], sys.argv[2]

with open(true_file) as f:
    reader = csv.DictReader(f)
    true_map = {}
    for i, r in enumerate(reader):
        true_map[i] = float(r.get("Rf", r.get("y_mean", 0)))

with open(pred_file) as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    pred_list = [float(r[cols[-1]]) for r in reader]

n = min(len(pred_list), len(true_map))
y_true = [true_map[i] for i in range(n)]
y_pred = pred_list[:n]

mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
mse = sum((t - p)**2 for t, p in zip(y_true, y_pred)) / n
rmse = math.sqrt(mse)
mean_t = sum(y_true) / n
ss_res = sum((t - p)**2 for t, p in zip(y_true, y_pred))
ss_tot = sum((t - mean_t)**2 for t in y_true)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

print("=" * 44)
print(f"  Evaluation Results (n={n})")
print("=" * 44)
print(f"  MAE:   {mae:.4f}")
print(f"  RMSE:  {rmse:.4f}")
print(f"  R²:    {r2:.4f}")
print(f"  True range:  [{min(y_true):.4f}, {max(y_true):.4f}]")
print(f"  Pred range:  [{min(y_pred):.4f}, {max(y_pred):.4f}]")
print("=" * 44)
PYEOF
