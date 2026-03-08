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
Distillation Pipeline Orchestrator — LangGraph StateGraph

Runs the full 5-step distillation pipeline with conditional edges that
verify each step completed successfully before proceeding to the next.

Each run gets a timestamped subfolder under runs/ so nothing is overwritten.
All paths are created once at init and carried in the state — no script
creates directories independently.

Usage:
    python run_pipeline.py                           # full run
    python run_pipeline.py --count 3 --epochs 1      # quick test
    python run_pipeline.py --run-id my-experiment    # custom run name
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    run_id: str
    run_dir: str
    paths: dict          # all sub-paths, created once at init
    teacher_model: str
    student_model: str
    count: int | None
    epochs: int
    lora_rank: int
    num_layers: int
    learning_rate: float
    batch_size: int
    max_tokens: int
    cost_estimate: bool
    error: str | None
    step_results: dict
    current_step: str


def create_run_directory(run_dir: str) -> dict:
    """Create the full directory tree once and return all paths."""
    root = Path(run_dir)

    paths = {
        "root":           str(root),
        "data":           str(root / "data"),
        "adapters":       str(root / "adapters"),
        "outputs_before": str(root / "outputs" / "before"),
        "outputs_after":  str(root / "outputs" / "after"),
        "outputs_teacher": str(root / "outputs" / "teacher"),
        "teacher_data":   str(root / "teacher_data.jsonl"),
        "train_data":     str(root / "data" / "train.jsonl"),
        "valid_data":     str(root / "data" / "valid.jsonl"),
        "lora_config":    str(root / "lora_config.yaml"),
        "comparison":     str(root / "outputs" / "comparison.html"),
        "cost_report":    str(root / "cost_report.json"),
        "run_config":     str(root / "run_config.json"),
        "run_results":    str(root / "run_results.json"),
    }

    # Create all directories upfront
    for dir_key in ["root", "data", "adapters", "outputs_before", "outputs_after", "outputs_teacher"]:
        Path(paths[dir_key]).mkdir(parents=True, exist_ok=True)

    return paths


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_extract_knowledge(state: PipelineState) -> dict:
    """Node 1: Extract knowledge from teacher model via API."""
    print("\n" + "=" * 70)
    print("  STEP 1/5: Extract Knowledge")
    print("=" * 70)

    start = time.time()
    try:
        from importlib import import_module
        mod = import_module("01_extract_knowledge")

        result = mod.generate_teacher_data(
            model=state["teacher_model"],
            count=state["count"],
            paths=state["paths"],
        )

        elapsed = time.time() - start
        results = dict(state["step_results"])

        success = result is not False
        step_info = {
            "success": success,
            "elapsed_seconds": round(elapsed, 1),
        }

        if success and isinstance(result, dict):
            step_info["cost"] = {
                "total_usd": result.get("total_cost_usd", 0),
                "input_tokens": result.get("total_input_tokens", 0),
                "output_tokens": result.get("total_output_tokens", 0),
                "examples": result.get("examples_generated", 0),
            }

        results["extract_knowledge"] = step_info

        if not success:
            return {"error": "Knowledge extraction failed — no examples produced", "step_results": results}

        teacher_file = Path(state["paths"]["teacher_data"])
        count = sum(1 for _ in open(teacher_file)) if teacher_file.exists() else 0
        results["extract_knowledge"]["examples"] = count

        return {"step_results": results, "current_step": "extract_knowledge"}

    except Exception as e:
        return {"error": f"Knowledge extraction crashed: {e}"}


def node_eval_baseline(state: PipelineState) -> dict:
    """Node 2: Evaluate baseline student model."""
    print("\n" + "=" * 70)
    print("  STEP 2/5: Evaluate Baseline")
    print("=" * 70)

    start = time.time()
    try:
        from importlib import import_module
        mod = import_module("02_eval_baseline")

        success = mod.run_baseline(
            model_name=state["student_model"],
            paths=state["paths"],
            max_tokens=state["max_tokens"],
        )

        elapsed = time.time() - start
        results = dict(state["step_results"])
        results["eval_baseline"] = {
            "success": success,
            "elapsed_seconds": round(elapsed, 1),
        }

        if not success:
            return {"error": "Baseline evaluation failed", "step_results": results}

        return {"step_results": results, "current_step": "eval_baseline"}

    except Exception as e:
        return {"error": f"Baseline evaluation crashed: {e}"}


def node_adapt_model(state: PipelineState) -> dict:
    """Node 3: Adapt student model via LoRA."""
    print("\n" + "=" * 70)
    print("  STEP 3/5: Adapt Model (LoRA)")
    print("=" * 70)

    start = time.time()
    try:
        from importlib import import_module
        mod = import_module("03_adapt_model")

        success = mod.finetune(
            model=state["student_model"],
            epochs=state["epochs"],
            lora_rank=state["lora_rank"],
            num_layers=state["num_layers"],
            learning_rate=state["learning_rate"],
            batch_size=state["batch_size"],
            paths=state["paths"],
        )

        elapsed = time.time() - start
        results = dict(state["step_results"])
        results["adapt_model"] = {
            "success": success,
            "elapsed_seconds": round(elapsed, 1),
        }

        if not success:
            return {"error": "Model adaptation failed", "step_results": results}

        return {"step_results": results, "current_step": "adapt_model"}

    except Exception as e:
        return {"error": f"Model adaptation crashed: {e}"}


def node_eval_adapted(state: PipelineState) -> dict:
    """Node 4: Evaluate adapted student model."""
    print("\n" + "=" * 70)
    print("  STEP 4/5: Evaluate Adapted Model")
    print("=" * 70)

    start = time.time()
    try:
        from importlib import import_module
        mod = import_module("04_eval_adapted")

        success = mod.run_distilled(
            model_name=state["student_model"],
            paths=state["paths"],
            max_tokens=state["max_tokens"],
        )

        elapsed = time.time() - start
        results = dict(state["step_results"])
        results["eval_adapted"] = {
            "success": success,
            "elapsed_seconds": round(elapsed, 1),
        }

        if not success:
            return {"error": "Adapted model evaluation failed — adapter not found?", "step_results": results}

        return {"step_results": results, "current_step": "eval_adapted"}

    except Exception as e:
        return {"error": f"Adapted model evaluation crashed: {e}"}


def node_benchmark(state: PipelineState) -> dict:
    """Node 5: Benchmark — before vs after comparison."""
    print("\n" + "=" * 70)
    print("  STEP 5/5: Benchmark")
    print("=" * 70)

    start = time.time()
    try:
        from importlib import import_module
        mod = import_module("05_benchmark")

        success = mod.generate_comparison(paths=state["paths"])

        elapsed = time.time() - start
        results = dict(state["step_results"])
        results["benchmark"] = {
            "success": success,
            "elapsed_seconds": round(elapsed, 1),
        }

        if not success:
            return {"error": "Benchmark generation failed", "step_results": results}

        return {"step_results": results, "current_step": "benchmark"}

    except Exception as e:
        return {"error": f"Benchmark crashed: {e}"}


def node_failed(state: PipelineState) -> dict:
    """Terminal node for failures."""
    print("\n" + "!" * 70)
    print("  PIPELINE FAILED")
    print("!" * 70)
    print(f"  Error: {state['error']}")
    print(f"  Run:   {state['run_dir']}")
    print()

    for step_name, step_data in state["step_results"].items():
        status = "✓" if step_data.get("success") else "✗"
        elapsed = step_data.get("elapsed_seconds", "?")
        print(f"  {status} {step_name} ({elapsed}s)")

    print()
    return {}


# ---------------------------------------------------------------------------
# Conditional edges — verify outputs using paths from state
# ---------------------------------------------------------------------------

def check_after_extraction(state: PipelineState) -> str:
    if state.get("error"):
        return "failed"

    p = state["paths"]
    teacher_file = Path(p["teacher_data"])
    train_file = Path(p["train_data"])

    if not teacher_file.exists():
        state["error"] = f"Verification failed: {teacher_file} does not exist"
        return "failed"

    count = sum(1 for _ in open(teacher_file))
    if count == 0:
        state["error"] = "Verification failed: teacher_data.jsonl is empty"
        return "failed"

    if not train_file.exists():
        state["error"] = f"Verification failed: {train_file} does not exist"
        return "failed"

    print(f"\n  ✓ Verified: {teacher_file} ({count} examples)")
    print(f"  ✓ Verified: {train_file} exists")
    return "continue"


def check_after_eval_baseline(state: PipelineState) -> str:
    if state.get("error"):
        return "failed"

    before_dir = Path(state["paths"]["outputs_before"])

    with open("scenarios/prompts.json") as f:
        expected = len(json.load(f)["test_prompts"])

    files = list(before_dir.glob("test_*.html")) + list(before_dir.glob("test_*.json")) + list(before_dir.glob("test_*.md"))
    if len(files) < expected:
        state["error"] = f"Verification failed: expected {expected} baseline files, found {len(files)}"
        return "failed"

    print(f"\n  ✓ Verified: {len(files)} baseline output files in {before_dir}")
    return "continue"


def check_after_adaptation(state: PipelineState) -> str:
    if state.get("error"):
        return "failed"

    adapter_dir = Path(state["paths"]["adapters"])

    if not adapter_dir.exists():
        state["error"] = f"Verification failed: {adapter_dir} does not exist"
        return "failed"

    adapter_files = list(adapter_dir.glob("*"))
    if not adapter_files:
        state["error"] = f"Verification failed: {adapter_dir} is empty"
        return "failed"

    print(f"\n  ✓ Verified: adapter saved at {adapter_dir} ({len(adapter_files)} files)")
    return "continue"


def check_after_eval_adapted(state: PipelineState) -> str:
    if state.get("error"):
        return "failed"

    after_dir = Path(state["paths"]["outputs_after"])

    with open("scenarios/prompts.json") as f:
        expected = len(json.load(f)["test_prompts"])

    files = list(after_dir.glob("test_*.html")) + list(after_dir.glob("test_*.json")) + list(after_dir.glob("test_*.md"))
    if len(files) < expected:
        state["error"] = f"Verification failed: expected {expected} adapted files, found {len(files)}"
        return "failed"

    print(f"\n  ✓ Verified: {len(files)} adapted output files in {after_dir}")
    return "continue"


def check_after_benchmark(state: PipelineState) -> str:
    if state.get("error"):
        return "failed"

    comparison = Path(state["paths"]["comparison"])

    if not comparison.exists():
        state["error"] = f"Verification failed: {comparison} does not exist"
        return "failed"

    print(f"\n  ✓ Verified: {comparison} exists ({comparison.stat().st_size:,} bytes)")
    return "done"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("extract_knowledge", node_extract_knowledge)
    graph.add_node("eval_baseline", node_eval_baseline)
    graph.add_node("adapt_model", node_adapt_model)
    graph.add_node("eval_adapted", node_eval_adapted)
    graph.add_node("benchmark", node_benchmark)
    graph.add_node("failed", node_failed)

    graph.set_entry_point("extract_knowledge")

    graph.add_conditional_edges(
        "extract_knowledge", check_after_extraction,
        {"continue": "eval_baseline", "failed": "failed"},
    )
    graph.add_conditional_edges(
        "eval_baseline", check_after_eval_baseline,
        {"continue": "adapt_model", "failed": "failed"},
    )
    graph.add_conditional_edges(
        "adapt_model", check_after_adaptation,
        {"continue": "eval_adapted", "failed": "failed"},
    )
    graph.add_conditional_edges(
        "eval_adapted", check_after_eval_adapted,
        {"continue": "benchmark", "failed": "failed"},
    )
    graph.add_conditional_edges(
        "benchmark", check_after_benchmark,
        {"done": END, "failed": "failed"},
    )

    graph.add_edge("failed", END)

    return graph


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the full distillation pipeline via LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                            # full run, all defaults
  python run_pipeline.py --count 3 --epochs 1       # quick smoke test
  python run_pipeline.py --run-id experiment-v2     # custom run name
  python run_pipeline.py --teacher-model claude-sonnet-4-20250514  # different API teacher
  python run_pipeline.py --teacher-model mlx-community/Qwen3.5-122B-A10B-bf16  # local MLX teacher (122B MoE)
  python run_pipeline.py --teacher-model mlx-community/Qwen3.5-27B-bf16         # local MLX teacher (27B dense)
        """,
    )
    parser.add_argument("--teacher-model", default="claude-opus-4-6",
                        help="Teacher model: Claude name for API, or HF path for local MLX (e.g. mlx-community/Qwen3.5-122B-A10B-bf16 or mlx-community/Qwen3.5-27B-bf16)")
    parser.add_argument("--student-model", default="mlx-community/Llama-3.1-8B-Instruct-4bit",
                        help="Student model for MLX (default: Llama 3.1 8B 4-bit — capable baseline, clear distillation improvement)")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of teacher examples (default: all 18)")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--run-id", default=None,
                        help="Custom run ID (default: timestamp)")
    parser.add_argument("--cost-estimate", action="store_true",
                        help="Generate a cost impact analysis XLSX after the run")

    args = parser.parse_args()

    # Create timestamped run directory with full tree — ONCE
    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = str(Path("runs") / run_id)
    paths = create_run_directory(run_dir)

    print()
    print("      ██╗ ██████╗ ███████╗     █████╗  ██████╗ ")
    print("      ██║██╔═══██╗██╔════╝    ██╔══██╗██╔═████╗")
    print("      ██║██║   ██║█████╗      ╚██████║██║██╔██║")
    print(" ██   ██║██║   ██║██╔══╝       ╚═══██║████╔╝██║")
    print(" ╚█████╔╝╚██████╔╝███████╗     █████╔╝╚██████╔╝")
    print("  ╚════╝  ╚═════╝ ╚══════╝     ╚════╝  ╚═════╝ ")
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  KNOWLEDGE EXTRACTION PIPELINE — LangGraph" + " " * 25 + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Run ID:    {run_id:<55} ║")
    print(f"║  Run Dir:   {run_dir:<55} ║")
    print(f"║  Teacher:   {args.teacher_model:<55} ║")
    print(f"║  Student:   {args.student_model:<55} ║")
    count_str = str(args.count) if args.count else "all"
    print(f"║  Examples:  {count_str:<55} ║")
    print(f"║  Epochs:    {args.epochs:<55} ║")
    print("╚" + "═" * 68 + "╝")

    # Save run config for reproducibility
    config = vars(args)
    config["run_id"] = run_id
    config["run_dir"] = run_dir
    config["paths"] = paths
    config["started_at"] = datetime.now().isoformat()
    with open(paths["run_config"], "w") as f:
        json.dump(config, f, indent=2)

    # Build and run the graph
    graph = build_graph()
    app = graph.compile()

    initial_state: PipelineState = {
        "run_id": run_id,
        "run_dir": run_dir,
        "paths": paths,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "count": args.count,
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "num_layers": args.num_layers,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "cost_estimate": args.cost_estimate,
        "error": None,
        "step_results": {},
        "current_step": "",
    }

    pipeline_start = time.time()
    final_state = app.invoke(initial_state)
    total_elapsed = time.time() - pipeline_start

    # Save final state
    final_state["finished_at"] = datetime.now().isoformat()
    final_state["total_elapsed_seconds"] = round(total_elapsed, 1)
    with open(paths["run_results"], "w") as f:
        json.dump(final_state, f, indent=2, default=str)

    # Print summary
    if final_state.get("error"):
        print(f"\n✗ Pipeline failed after {total_elapsed:.0f}s")
        print(f"  Error: {final_state['error']}")
        exit(1)
    else:
        print("\n" + "=" * 70)
        print("  PIPELINE COMPLETE")
        print("=" * 70)
        print(f"  Total time: {total_elapsed:.0f}s")
        print(f"  Run dir:    {run_dir}/")
        print()
        for step_name, step_data in final_state["step_results"].items():
            elapsed = step_data.get("elapsed_seconds", "?")
            print(f"  ✓ {step_name} ({elapsed}s)")
        print()

        # Cost summary
        teacher_step = final_state["step_results"].get("extract_knowledge", {})
        cost_data = teacher_step.get("cost")
        if cost_data:
            is_local = "/" in final_state.get("teacher_model", "")
            print(f"  {'─' * 40}")
            if is_local:
                print(f"  🖥  Local Teacher Summary")
            else:
                print(f"  💰 Anthropic API Cost")
            print(f"  {'─' * 40}")
            print(f"  Input tokens:  {cost_data['input_tokens']:>10,}")
            print(f"  Output tokens: {cost_data['output_tokens']:>10,}")
            if not is_local:
                print(f"  Total cost:    ${cost_data['total_usd']:>9.4f}")
            print(f"  {'─' * 40}")
            print(f"  Full breakdown: {paths['cost_report']}")
            print()

        print(f"  Open results: open {paths['comparison']}")

        # Cost estimate report
        if final_state.get("cost_estimate"):
            try:
                from cost_estimate import generate_cost_estimate
                xlsx_path = generate_cost_estimate(
                    run_dir=run_dir,
                    run_results=final_state,
                )
                print(f"  Open estimate: open {xlsx_path}")
            except Exception as e:
                print(f"\n  ⚠ Cost estimate generation failed: {e}")


if __name__ == "__main__":
    main()
