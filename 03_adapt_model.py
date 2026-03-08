#!/usr/bin/env python3
# Copyright 2025 Nicolas Cravino
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Step 3: Adapt Student Model via LoRA (MLX)

Fine-tunes the student model on extracted knowledge using
Low-Rank Adaptation (LoRA) via mlx-lm.

Usage:
    python 03_adapt_model.py
    python 03_adapt_model.py --epochs 6 --run-dir runs/my-test
"""

import argparse
import math
import subprocess
import sys
import yaml
from pathlib import Path


def _make_default_paths(run_dir: str) -> dict:
    """Build a paths dict for standalone usage (not via pipeline)."""
    root = Path(run_dir)
    return {
        "root":        str(root),
        "data":        str(root / "data"),
        "adapters":    str(root / "adapters"),
        "train_data":  str(root / "data" / "train.jsonl"),
        "valid_data":  str(root / "data" / "valid.jsonl"),
        "lora_config": str(root / "lora_config.yaml"),
    }


def check_data_exists(train_data_path: str) -> int:
    """Verify training data is ready. Returns count or raises."""
    train_file = Path(train_data_path)
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found at {train_file}")

    count = sum(1 for _ in open(train_file))
    print(f"  Training examples: {count}")
    return count


def finetune(
    model: str = "mlx-community/Llama-3.1-8B-Instruct-4bit",
    iters: int | None = None,
    epochs: int = 4,
    lora_rank: int = 64,
    num_layers: int = 16,
    learning_rate: float = 2e-4,
    batch_size: int = 1,
    paths: dict | None = None,
    run_dir: str = ".",
) -> bool:
    """Fine-tune with LoRA. Returns True on success, False on failure."""
    if paths is None:
        paths = _make_default_paths(run_dir)

    adapter_path = Path(paths["adapters"])
    data_path = Path(paths["data"])
    config_path = Path(paths["lora_config"])

    print("=" * 60)
    print("  MLX LoRA Fine-Tuning (Distillation)")
    print("=" * 60)
    print(f"  Student model:  {model}")
    print(f"  LoRA rank:      {lora_rank}")
    print(f"  LoRA layers:    {num_layers}")
    print(f"  Epochs:         {epochs}")
    print(f"  Learning rate:  {learning_rate}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Run directory:  {paths['root']}")
    print(f"  Adapter output: {adapter_path}/")
    print()

    try:
        num_examples = check_data_exists(paths["train_data"])
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return False

    if iters is None:
        iters = math.ceil(num_examples / batch_size) * epochs
        print(f"  Calculated iters: {iters} ({epochs} epochs × {math.ceil(num_examples / batch_size)} steps/epoch)")

    config = {
        "model": model,
        "train": True,
        "data": str(data_path),
        "adapter_path": str(adapter_path),
        "iters": iters,
        "batch_size": batch_size,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "fine_tune_type": "lora",
        "lora_parameters": {
            "rank": lora_rank,
            "alpha": lora_rank,
            "dropout": 0.05,
            "scale": 1.0,
        },
        "save_every": max(iters // 4, 1),
        "steps_per_report": 1,
        "steps_per_eval": max(iters // 4, 1),
        "max_seq_length": 8192,
        "seed": 42,
    }

    if Path(paths["valid_data"]).exists():
        config["val_batches"] = 2

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n  Config written to {config_path}")
    print("\nStarting LoRA fine-tuning...\n")

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "-c", str(config_path),
    ]

    print(f"Running: {' '.join(cmd)}\n")

    process = subprocess.run(cmd, text=True)

    if process.returncode != 0:
        print(f"\n✗ Training failed with exit code {process.returncode}")
        return False

    print(f"\n✓ Fine-tuning complete! Adapter saved to {adapter_path}/")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune student model with LoRA")
    parser.add_argument("--model", default="mlx-community/Llama-3.1-8B-Instruct-4bit")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--run-dir", default=".")
    args = parser.parse_args()

    success = finetune(
        model=args.model, iters=args.iters, epochs=args.epochs,
        lora_rank=args.lora_rank, num_layers=args.num_layers,
        learning_rate=args.learning_rate, batch_size=args.batch_size,
        run_dir=args.run_dir,
    )
    if not success:
        exit(1)
