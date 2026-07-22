#!/usr/bin/env bash
set -euo pipefail

TRAIN_CROP_SIZE=2000
DIFFUSION_BATCH_SIZE=10
DIFFUSION_CHUNK_SIZE=4

STEPS="${1:-0}"
if [[ "${STEPS}" -eq 0 ]]; then
  EVAL_ONLY=true
  MODE=eval
else
  EVAL_ONLY=false
  MODE=train
fi

RUN_NAME="lynxkite_${MODE}_steps${STEPS}"
BASE_DIR="./output_${MODE}_steps${STEPS}"

export BIOKINEMA_INIT_CKPT=/home/daniel/BioKinema/checkpoints/BioKinema_atlas+misato+mdposit_sqrt.pt
export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTTENTION=true
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CUDA_VISIBLE_DEVICES=0 uv run python runner/train.py \
  --run_name "${RUN_NAME}" \
  --project protenix \
  --base_dir "${BASE_DIR}" \
  --dtype bf16 \
  --train_crop_size "${TRAIN_CROP_SIZE}" \
  --diffusion_batch_size "${DIFFUSION_BATCH_SIZE}" \
  --diffusion_chunk_size "${DIFFUSION_CHUNK_SIZE}" \
  --use_wandb false \
  --eval_only ${EVAL_ONLY} \
  --max_steps ${STEPS} \
  --load_checkpoint_path "${BIOKINEMA_INIT_CKPT}" \
  --load_params_only true \
  --data.train_sets lynxkite_train \
  --data.test_sets lynxkite_test \
  --data.num_dl_workers 1 \
  --data.msa.enable false \
  --eval_interval 100 \
  --log_interval 10
