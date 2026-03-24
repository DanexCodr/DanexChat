#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM, AutoTokenizer


def prune_linear_layers(model: torch.nn.Module, amount: float) -> None:
    modules_to_prune = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            modules_to_prune.append((module, "weight"))

    if not modules_to_prune:
        raise RuntimeError("No linear layers found to prune")

    prune.global_unstructured(
        modules_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    for module, _ in modules_to_prune:
        prune.remove(module, "weight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune a CausalLM checkpoint with global magnitude pruning")
    parser.add_argument("--base-model", required=True, help="HF model id or local checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sparsity", type=float, default=0.2, help="0.0-1.0 global sparsity")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    if not (0.0 <= args.sparsity < 1.0):
        raise ValueError("--sparsity must be in [0.0, 1.0)")

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    prune_linear_layers(model, args.sparsity)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved pruned model to {args.output_dir}")


if __name__ == "__main__":
    main()
