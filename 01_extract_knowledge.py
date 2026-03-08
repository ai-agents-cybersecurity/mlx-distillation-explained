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
Step 1: Extract Knowledge from Teacher Model

Supports two backends:
  - Anthropic API (claude-*) — requires ANTHROPIC_API_KEY
  - Local MLX model (mlx-vlm) — any HuggingFace model path with '/'

Usage:
    python 01_extract_knowledge.py
    python 01_extract_knowledge.py --count 5
    python 01_extract_knowledge.py --run-dir runs/my-test
    python 01_extract_knowledge.py --model mlx-community/Qwen3.5-122B-A10B-bf16
    python 01_extract_knowledge.py --model mlx-community/Qwen3.5-27B-bf16
"""

import argparse
import json
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _is_local_model(model: str) -> bool:
    """Return True if model is a local HuggingFace/MLX path, False if Anthropic API."""
    return "/" in model

# Pricing per million tokens (USD) — Anthropic API only
MODEL_PRICING = {
    "claude-sonnet-4-6":          {"input": 3.00,  "output": 15.00},
    "claude-sonnet-4-5":          {"input": 3.00,  "output": 15.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00,  "output": 15.00},
    "claude-sonnet-4-20250514":   {"input": 3.00,  "output": 15.00},
    "claude-opus-4-6":            {"input": 5.00,  "output": 25.00},
    "claude-opus-4-5":            {"input": 5.00,  "output": 25.00},
    "claude-opus-4-5-20251101":   {"input": 5.00,  "output": 25.00},
    "claude-haiku-4-5":           {"input": 1.00,  "output": 5.00},
    "claude-haiku-4-5-20251001":  {"input": 1.00,  "output": 5.00},
}
DEFAULT_PRICING = {"input": 3.00, "output": 15.00}


# ---------------------------------------------------------------------------
# Local MLX teacher (mlx-vlm)
# ---------------------------------------------------------------------------

_mlx_model_cache: dict = {}


def _load_mlx_teacher(model_path: str):
    """Lazy-load an MLX-VLM model. Cached so repeated calls are free."""
    if model_path in _mlx_model_cache:
        return _mlx_model_cache[model_path]

    from mlx_vlm.utils import load_config

    print(f"  Loading local MLX teacher: {model_path} ...")

    # Try standard load; fall back to manual assembly on TypeError
    # (works around mlx-vlm bugs where StoppingCriteria receives None eos_token_ids)
    try:
        from mlx_vlm import load
        model, processor = load(model_path)
    except TypeError:
        print("  ⚠ mlx_vlm.load() failed, using fallback loader...")
        model, processor = _load_mlx_fallback(model_path)

    config = load_config(model_path)
    _mlx_model_cache[model_path] = (model, processor, config)
    return model, processor, config


def _load_mlx_fallback(model_path_str: str):
    """Manual model+tokenizer assembly for text-only generation.

    Bypasses AutoProcessor (which triggers a transformers bug in
    video_processing_auto.py for Qwen3.5) and uses AutoTokenizer directly.
    """
    from mlx_vlm.utils import (
        get_model_path, load_model, load_config,
        StoppingCriteria,
    )
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from transformers import AutoTokenizer

    resolved = get_model_path(model_path_str)
    model = load_model(resolved)
    config = load_config(resolved)

    # Read eos_token_id from config dict (always present)
    eos_ids = config.get("eos_token_id", None)
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    elif eos_ids is None:
        eos_ids = []

    # Use AutoTokenizer instead of AutoProcessor (text-only, no vision/video needed)
    tokenizer = AutoTokenizer.from_pretrained(resolved)

    # Attach detokenizer (needed by mlx_vlm.generate for streaming decode)
    detokenizer_class = load_tokenizer(resolved, return_tokenizer=False)
    tokenizer.detokenizer = detokenizer_class(tokenizer)

    # Attach stopping criteria with safe (non-None) eos_ids
    tokenizer.stopping_criteria = StoppingCriteria(eos_ids, tokenizer)

    return model, tokenizer


def _generate_mlx(model_path: str, system_prompt: str, user_prompt: str,
                  max_tokens: int = 8192) -> dict:
    """Generate a single completion with a local MLX-VLM model.

    Returns dict with keys: completion, prompt_tokens, generation_tokens.
    """
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, config = _load_mlx_teacher(model_path)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    formatted_prompt = apply_chat_template(
        processor, config, messages, num_images=0,
    )

    result = generate(
        model, processor, formatted_prompt,
        image=None, verbose=False, max_tokens=max_tokens,
    )

    return {
        "completion": result.text,
        "prompt_tokens": result.prompt_tokens,
        "generation_tokens": result.generation_tokens,
    }


def _make_default_paths(run_dir: str) -> dict:
    """Build a paths dict for standalone usage (not via pipeline)."""
    root = Path(run_dir)
    return {
        "root":           str(root),
        "data":           str(root / "data"),
        "teacher_data":   str(root / "teacher_data.jsonl"),
        "train_data":     str(root / "data" / "train.jsonl"),
        "valid_data":     str(root / "data" / "valid.jsonl"),
        "cost_report":    str(root / "cost_report.json"),
    }


def generate_teacher_data(
    model: str = "claude-sonnet-4-6",
    count: int | None = None,
    paths: dict | None = None,
    run_dir: str = ".",
) -> bool | dict:
    """Generate teacher data. Returns cost_info dict on success, False on failure."""
    if paths is None:
        paths = _make_default_paths(run_dir)

    # Ensure dirs exist (no-op if pipeline already created them)
    Path(paths["data"]).mkdir(parents=True, exist_ok=True)

    local = _is_local_model(model)

    if local:
        # Pre-load model once (fail fast instead of retrying per prompt)
        try:
            _load_mlx_teacher(model)
        except Exception as e:
            print(f"\n✗ Failed to load local MLX teacher: {model}")
            traceback.print_exc()
            return False
    else:
        import anthropic
        client = anthropic.Anthropic()
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)

    with open("scenarios/prompts.json") as f:
        prompts = json.load(f)

    system_prompt = prompts["system_prompt"]
    training_prompts = prompts["training_prompts"]

    if count is not None:
        training_prompts = training_prompts[:count]

    total = len(training_prompts)
    backend = "local MLX" if local else "Anthropic API"
    print(f"Generating training data from {model} ({backend})...")
    print(f"  System prompt: {system_prompt[:80]}...")
    print(f"  Training examples: {total}")
    if not local:
        print(f"  Pricing: ${pricing['input']}/MTok input, ${pricing['output']}/MTok output")
    print(f"  Output: {paths['teacher_data']}\n")

    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    per_request_costs = []

    for i, prompt in enumerate(training_prompts):
        short = prompt[:75] + "..." if len(prompt) > 75 else prompt
        print(f"[{i+1}/{total}] {short}")

        try:
            if local:
                # ── Local MLX teacher ──
                mlx_result = _generate_mlx(
                    model, system_prompt, prompt, max_tokens=8192,
                )
                completion = mlx_result["completion"]
                inp_tok = mlx_result["prompt_tokens"]
                out_tok = mlx_result["generation_tokens"]
                req_cost = 0.0
            else:
                # ── Anthropic API teacher ──
                response = client.messages.create(
                    model=model,
                    max_tokens=8192,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )

                completion = response.content[0].text

                usage = response.usage
                inp_tok = usage.input_tokens
                out_tok = usage.output_tokens
                req_cost = (inp_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000

            total_input_tokens += inp_tok
            total_output_tokens += out_tok
            total_cost += req_cost
            per_request_costs.append({
                "prompt_index": i,
                "input_tokens": inp_tok,
                "output_tokens": out_tok,
                "cost_usd": round(req_cost, 6),
            })

            completion = completion.strip()
            # Strip outer markdown fences if model wrapped the entire response
            if completion.startswith("```html"):
                completion = completion[7:]
                if completion.endswith("```"):
                    completion = completion[:-3]
                completion = completion.strip()
            elif completion.startswith("```"):
                completion = completion[3:]
                if completion.endswith("```"):
                    completion = completion[:-3]
                completion = completion.strip()

            results.append({
                "system": system_prompt,
                "prompt": prompt,
                "completion": completion,
            })

            if local:
                print(f"  ✓ Got {len(completion)} chars | {inp_tok}+{out_tok} tokens")
            else:
                print(f"  ✓ Got {len(completion)} chars | {inp_tok}+{out_tok} tokens | ${req_cost:.4f} (total: ${total_cost:.4f})")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            traceback.print_exc()
            continue

        if not local:
            time.sleep(0.5)

    if not results:
        print("\n✗ No training examples were generated.")
        return False

    # Cost / token summary
    cost_info = {
        "model": model,
        "backend": backend,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_cost_usd": round(total_cost, 4),
        "avg_cost_per_example_usd": round(total_cost / len(results), 4) if total_cost > 0 else 0,
        "examples_generated": len(results),
        "per_request": per_request_costs,
    }

    print(f"\n{'─' * 50}")
    if local:
        print(f"  �  Local Generation Summary")
    else:
        print(f"  �� Cost Summary")
    print(f"{'─' * 50}")
    print(f"  Input tokens:  {total_input_tokens:>10,}")
    print(f"  Output tokens: {total_output_tokens:>10,}")
    print(f"  Total tokens:  {total_input_tokens + total_output_tokens:>10,}")
    if not local:
        print(f"  Total cost:    ${total_cost:>9.4f}")
        print(f"  Per example:   ${total_cost / len(results):>9.4f}")
    print(f"{'─' * 50}")

    with open(paths["cost_report"], "w") as f:
        json.dump(cost_info, f, indent=2)
    print(f"  Saved to {paths['cost_report']}")

    # Save JSONL
    with open(paths["teacher_data"], "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    print(f"\n✓ Saved {len(results)} training examples to {paths['teacher_data']}")

    # Save MLX chat format
    train_records = []
    for item in results:
        train_records.append({
            "messages": [
                {"role": "system", "content": item["system"]},
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["completion"]},
            ]
        })

    with open(paths["train_data"], "w") as f:
        for rec in train_records:
            f.write(json.dumps(rec) + "\n")

    valid_count = min(5, len(train_records))
    with open(paths["valid_data"], "w") as f:
        for rec in train_records[-valid_count:]:
            f.write(json.dumps(rec) + "\n")

    print(f"✓ Saved MLX data to {paths['train_data']} ({len(train_records)} train, {valid_count} valid)")

    # --- Generate teacher outputs for test prompts (used in 3-way comparison) ---
    teacher_output_dir = paths.get("outputs_teacher")
    if teacher_output_dir:
        test_prompts = prompts.get("test_prompts", [])
        if test_prompts:
            Path(teacher_output_dir).mkdir(parents=True, exist_ok=True)
            print(f"\nGenerating teacher outputs for {len(test_prompts)} test prompts...")

            for i, prompt in enumerate(test_prompts):
                short = prompt[:75] + "..." if len(prompt) > 75 else prompt
                print(f"  [test {i+1}/{len(test_prompts)}] {short}")

                try:
                    if local:
                        mlx_result = _generate_mlx(model, system_prompt, prompt, max_tokens=8192)
                        completion = mlx_result["completion"]
                    else:
                        response = client.messages.create(
                            model=model, max_tokens=8192, system=system_prompt,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        completion = response.content[0].text

                    completion = completion.strip()
                    # Strip outer markdown fences if model wrapped the entire response
                    if completion.startswith("```json"):
                        completion = completion[7:]
                        if completion.endswith("```"):
                            completion = completion[:-3]
                        completion = completion.strip()
                    elif completion.startswith("```"):
                        completion = completion[3:]
                        if completion.endswith("```"):
                            completion = completion[:-3]
                        completion = completion.strip()

                    ext = ".html"
                    out_file = Path(teacher_output_dir) / f"test_{i+1}{ext}"
                    with open(out_file, "w") as f:
                        f.write(completion)
                    print(f"    ✓ Saved {len(completion)} chars → {out_file}")

                except Exception as e:
                    print(f"    ✗ Error: {e}")

                if not local:
                    time.sleep(0.5)

            print(f"✓ Teacher test outputs saved to {teacher_output_dir}/")

    return cost_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate teacher data (Anthropic API or local MLX)")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Claude model name (API) or HuggingFace model path (local MLX)")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--run-dir", default=".")
    args = parser.parse_args()

    result = generate_teacher_data(model=args.model, count=args.count, run_dir=args.run_dir)
    if not result:
        exit(1)
