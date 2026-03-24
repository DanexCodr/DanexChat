# Training pipeline (prune + chat-style fine-tune)

This directory adds an offline retraining workflow for DanexChat's base model.

## What it does

1. Builds chat-style supervised training rows in ChatML format.
2. Applies global unstructured pruning to the base CausalLM.
3. Fine-tunes the pruned checkpoint on tag-oriented chat examples.

Target tags in training data:
- `$date`
- `$key`
- `$value`
- `$result`

## Files

- `requirements.txt` — Python dependencies for this training workflow
- `data/chat_style_tag_supervised.jsonl` — chat-style tag examples
- `data/tag_supervised.jsonl` — simple prompt/response tag examples
- `prepare_chat_dataset.py` — converts JSONL records into ChatML `{"text": ...}` rows
- `prune_model.py` — global magnitude pruning for linear layers
- `finetune_chat_sft.py` — supervised fine-tuning with Hugging Face Trainer
- `run_pipeline.sh` — one-command prune + fine-tune pipeline

## Setup

```bash
cd /home/runner/work/DanexChat/DanexChat
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r training/requirements.txt
```

## Run end-to-end pipeline

```bash
cd /home/runner/work/DanexChat/DanexChat
bash training/run_pipeline.sh \
  HuggingFaceTB/SmolLM2-135M-Instruct \
  0.20 \
  training/artifacts
```

Arguments:
1. Base model id/path (default: `HuggingFaceTB/SmolLM2-135M-Instruct`)
2. Global sparsity amount (default: `0.20`)
3. Output workdir (default: `training/artifacts`)

Outputs:
- Pruned checkpoint: `training/artifacts/pruned`
- Prepared training data: `training/artifacts/data/chatml_train.jsonl`
- Fine-tuned checkpoint: `training/artifacts/sft`

## Optional: run steps individually

```bash
python training/prepare_chat_dataset.py \
  --input training/data/chat_style_tag_supervised.jsonl \
  --output training/artifacts/data/chatml_train.jsonl

python training/prune_model.py \
  --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --output-dir training/artifacts/pruned \
  --sparsity 0.20 \
  --dtype float16

python training/finetune_chat_sft.py \
  --model training/artifacts/pruned \
  --train-file training/artifacts/data/chatml_train.jsonl \
  --output-dir training/artifacts/sft \
  --epochs 2 \
  --lr 2e-5 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-length 768 \
  --dtype float16
```

## Notes

- `onnx` was intentionally not added as a required dependency because currently known advisories affect available versions in the advisory DB context.
- Exporting the fine-tuned checkpoint to ONNX for Android runtime can be added in a follow-up once a safe/approved export stack is selected.
