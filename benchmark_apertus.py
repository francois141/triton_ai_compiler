"""Compile Apertus kernels and compare injected-PTX throughput with baseline."""

import argparse
import json
from pathlib import Path

import torch

from agent import run_agent_loop
from triton_ptx.LLMs import measure
from triton_ptx.LLMs.apertus import Apertus
from utils.response_format import PtxKernel

MESSAGES = [
    {
        "role": "system",
        "content": "You are a concise and helpful assistant.",
    },
    {
        "role": "user",
        "content": "Explain why GPUs are efficient at parallel computation.",
    },
]

def load_verified_payload(trace_directory):
    """Load the fastest verified PTX payload saved in one trace directory."""
    candidates = []
    final_artifacts = trace_directory.glob("final_speedup_vs_triton_*.json")
    candidate_artifacts = trace_directory.glob("iteration_*_speedup_vs_triton_*.json")
    for artifact_path in (*final_artifacts, *candidate_artifacts):
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        candidate = artifact.get("candidate", artifact)
        evaluation = artifact.get("evaluation", artifact)
        if not evaluation.get("compiles") or not evaluation.get("correct"):
            continue
        p50 = evaluation.get("p50", artifact.get("p50"))
        if p50 is None:
            continue
        candidates.append((p50, candidate))
    if not candidates:
        raise ValueError(f"No verified payload found in {trace_directory}.")
    return PtxKernel.model_validate(min(candidates, key=lambda item: item[0])[1])


def recent_trace_payloads(llm_class, trace_path, trace_count):
    """Return verified payloads from the most recently modified trace folders."""
    if trace_count == 0 or not trace_path.exists():
        return {}
    recent_directories = sorted(
        (path for path in trace_path.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    trace_directories = []
    for trace_directory in recent_directories:
        try:
            load_verified_payload(trace_directory)
        except ValueError:
            continue
        trace_directories.append(trace_directory)
        if len(trace_directories) == trace_count:
            break
    payloads = {}
    for name, kernel_class in llm_class.get_kernel_classes().items():
        matching_traces = [
            path
            for path in trace_directories
            if f"_{kernel_class.__name__}_" in path.name
        ]
        if not matching_traces:
            continue
        trace_directory = matching_traces[0]
        payloads[f"{name}_payload"] = load_verified_payload(trace_directory)
        print(f"Reused verified {name} from {trace_directory}.")
    return payloads


def generate_kernel_payloads(llm_class, arguments):
    """Generate and verify PTX for every Apertus kernel."""
    payloads = recent_trace_payloads(
        llm_class,
        arguments.trace_path,
        arguments.reuse_recent_traces,
    )
    for name, kernel_class in llm_class.get_kernel_classes().items():
        payload_name = f"{name}_payload"
        if payload_name in payloads:
            continue
        result = run_agent_loop(
            kernel_class.__name__,
            model=arguments.agent_model,
            provider=arguments.provider,
            max_tool_rounds=1,
            max_repair_attempts=arguments.max_repair_attempts,
            reasoning_effort=arguments.reasoning_effort,
            trace_path=arguments.trace_path,
        )
        candidate_data = json.loads(result)
        payloads[payload_name] = PtxKernel.model_validate(
            {
                field_name: candidate_data[field_name]
                for field_name in PtxKernel.model_fields
                if field_name in candidate_data
            }
        )
        print(f"Generated and verified {name} without improvement rounds.")
    return payloads


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--agent-model", default="gpt-5.6-sol")
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic"),
        default="openai",
    )
    parser.add_argument("--max-repair-attempts", type=int, default=10)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--trace-path", type=Path, default=Path("output_traces"))
    parser.add_argument(
        "--reuse-recent-traces",
        type=int,
        default=7,
        help="Reuse verified payloads from this many recent trace folders (default: 4).",
    )
    parser.add_argument(
        "--disable-ptx-linear",
        action="store_true",
        help="Use autotuned Triton for linear instead of its injected PTX.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")
    if arguments.reuse_recent_traces < 0:
        raise ValueError("--reuse-recent-traces must be non-negative.")
    payloads = generate_kernel_payloads(Apertus, arguments)
    if arguments.disable_ptx_linear:
        payloads.pop("linear_payload", None)
    optimized = Apertus.from_custom_ptx(**payloads)
    _, optimized_tokens_per_second = measure(
        optimized,
        MESSAGES,
        max_new_tokens=arguments.max_new_tokens,
    )
    tuned_triton = Apertus()
    _, tuned_triton_tokens_per_second = measure(
        tuned_triton,
        MESSAGES,
        max_new_tokens=arguments.max_new_tokens,
    )
    del tuned_triton
    torch.cuda.empty_cache()
    speedup = optimized_tokens_per_second / tuned_triton_tokens_per_second
    print(f"Autotuned Triton: {tuned_triton_tokens_per_second:.2f} tokens/s")
    injected_label = "Injected PTX"
    if arguments.disable_ptx_linear:
        injected_label += " (autotuned linear)"
    print(f"{injected_label}: {optimized_tokens_per_second:.2f} tokens/s")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
