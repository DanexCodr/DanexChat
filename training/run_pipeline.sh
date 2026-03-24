#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${1:-HuggingFaceTB/SmolLM2-135M-Instruct}"
SPARSITY="${2:-0.20}"
WORKDIR="${3:-training/artifacts}"

PRUNED_DIR="${WORKDIR}/pruned"
DATA_DIR="${WORKDIR}/data"
SFT_DIR="${WORKDIR}/sft"

mkdir -p "${WORKDIR}" "${DATA_DIR}"

python training/prepare_chat_dataset.py \
  --input training/data/chat_style_tag_supervised.jsonl \
  --output "${DATA_DIR}/chatml_train.jsonl"

python training/prune_model.py \
  --base-model "${BASE_MODEL}" \
  --output-dir "${PRUNED_DIR}" \
  --sparsity "${SPARSITY}" \
  --dtype float16

python training/finetune_chat_sft.py \
  --model "${PRUNED_DIR}" \
  --train-file "${DATA_DIR}/chatml_train.jsonl" \
  --output-dir "${SFT_DIR}" \
  --epochs 2 \
  --lr 2e-5 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-length 768 \
  --dtype float16

echo "Pipeline complete. Fine-tuned model: ${SFT_DIR}"
