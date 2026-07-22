#!/usr/bin/env bash
set -euo pipefail

STEPS="${1:-0}"
if [[ "${STEPS}" -eq 0 ]]; then
  EVAL_ONLY=true
else
  EVAL_ONLY=false
fi

export BIOKINEMA_INIT_CKPT=/home/daniel/BioKinema/checkpoints/BioKinema_atlas+misato+mdposit_sqrt.pt
export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTTENTION=true
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 uv run python runner/train.py \
  --run_name lynxkite_eval_only \
  --project protenix \
  --base_dir ./output_eval \
  --dtype bf16 \
  --train_crop_size 1000 \
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
