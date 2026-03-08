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
Step 4: Evaluate Adapted Student Model

Runs the same test prompts with the LoRA adapter loaded to measure
post-adaptation quality.

Usage:
    python 04_eval_adapted.py
    python 04_eval_adapted.py --run-dir runs/my-test
"""

import argparse
import json
from pathlib import Path

from mlx_lm import load, generate


def _make_default_paths(run_dir: str) -> dict:
    """Build a paths dict for standalone usage (not via pipeline)."""
    root = Path(run_dir)
    return {
        "root":          str(root),
        "adapters":      str(root / "adapters"),
        "outputs_after": str(root / "outputs" / "after"),
    }


def run_distilled(
    model_name: str = "mlx-community/Llama-3.1-8B-Instruct-4bit",
    paths: dict | None = None,
    run_dir: str = ".",
    max_tokens: int = 4096,
) -> bool:
    """Run distilled inference. Returns True on success."""
    if paths is None:
        paths = _make_default_paths(run_dir)

    adapter_path = Path(paths["adapters"])
    output_path = Path(paths["outputs_after"])
    output_path.mkdir(parents=True, exist_ok=True)

    if not adapter_path.exists():
        print(f"✗ Adapter not found at {adapter_path}/")
        return False

    with open("scenarios/prompts.json") as f:
        prompts = json.load(f)

    system_prompt = prompts["system_prompt"]
    test_prompts = prompts["test_prompts"]

    print(f"Student model:   {model_name}")
    print(f"LoRA adapter:    {adapter_path}/")
    print(f"Test prompts:    {len(test_prompts)}")
    print(f"Output dir:      {output_path}\n")

    print("Loading model with LoRA adapter...")
    model, tokenizer = load(model_name, adapter_path=str(adapter_path))

    for i, prompt in enumerate(test_prompts):
        short = prompt[:70] + "..." if len(prompt) > 70 else prompt
        print(f"\n[{i+1}/{len(test_prompts)}] {short}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            verbose=False,
        )

        output = response.strip()
        # Strip outer markdown fences if model wrapped the entire response
        if output.startswith("```html"):
            output = output[7:]
            if output.endswith("```"):
                output = output[:-3]
            output = output.strip()
        elif output.startswith("```"):
            output = output[3:]
            if output.endswith("```"):
                output = output[:-3]
            output = output.strip()

        # All outputs are HTML landing pages
        ext = ".html"
        out_file = output_path / f"test_{i+1}{ext}"
        with open(out_file, "w") as f:
            f.write(output)

        print(f"  ✓ Saved {len(output)} chars → {out_file}")

    print(f"\n✓ Distilled inference complete. Outputs in {output_path}/")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run distilled inference with LoRA adapter")
    parser.add_argument("--model", default="mlx-community/Llama-3.1-8B-Instruct-4bit")
    parser.add_argument("--run-dir", default=".")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    success = run_distilled(model_name=args.model, run_dir=args.run_dir, max_tokens=args.max_tokens)  # standalone: uses _make_default_paths
    if not success:
        exit(1)
