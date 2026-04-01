#!/usr/bin/env python3
"""
Export prajjwal1/bert-tiny to ONNX and extract the vocabulary file for DanexChat.

Usage:
    pip install torch transformers onnx onnxruntime
    python export_bert_tiny.py --output-dir bert_tiny_assets

Then copy the contents of bert_tiny_assets/ into:
    app/src/main/assets/bert_tiny/
"""

import argparse
import os

import onnx
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "prajjwal1/bert-tiny"


def export_bert_tiny(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # ------------------------------------------------------------------ vocab
    vocab_path = os.path.join(output_dir, "bert_vocab.txt")
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, _ in sorted_vocab:
            f.write(token + "\n")
    print(f"Vocabulary saved  → {vocab_path}  ({len(sorted_vocab)} tokens)")

    # ------------------------------------------------------------------ ONNX
    onnx_path = os.path.join(output_dir, "bert_tiny.onnx")
    dummy = tokenizer("Hello world", return_tensors="pt")
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]
    token_type_ids = dummy.get("token_type_ids", torch.zeros_like(input_ids))

    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask, token_type_ids),
            onnx_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids":        {0: "batch", 1: "seq"},
                "attention_mask":   {0: "batch", 1: "seq"},
                "token_type_ids":   {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
                "pooler_output":    {0: "batch"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

    # Verify the exported model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    size_mb = os.path.getsize(onnx_path) / 1_000_000
    print(f"ONNX model saved  → {onnx_path}  ({size_mb:.1f} MB)")

    print()
    print("Next steps:")
    print(f"  Copy {output_dir}/bert_tiny.onnx  →  app/src/main/assets/bert_tiny/bert_tiny.onnx")
    print(f"  Copy {output_dir}/bert_vocab.txt   →  app/src/main/assets/bert_tiny/bert_vocab.txt")
    print()
    print("BertTinyEncoder will automatically load these at app startup.")
    print("Without them the app falls back to keyword-based routing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export BERT-tiny to ONNX for use in DanexChat")
    parser.add_argument(
        "--output-dir",
        default="bert_tiny_assets",
        help="Directory to write bert_tiny.onnx and bert_vocab.txt (default: bert_tiny_assets)")
    args = parser.parse_args()
    export_bert_tiny(args.output_dir)
