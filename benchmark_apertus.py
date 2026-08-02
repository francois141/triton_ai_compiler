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


def generate_kernel_payloads(llm_class, arguments):
    """Generate and verify PTX for every Apertus kernel."""
    payloads = {}
    for name, kernel_class in llm_class.get_kernel_classes().items():
        result = run_agent_loop(
            kernel_class.__name__,
            model=arguments.agent_model,
            provider=arguments.provider,
            max_tool_rounds=0,
            max_repair_attempts=arguments.max_repair_attempts,
            reasoning_effort=arguments.reasoning_effort,
            trace_path=arguments.trace_path,
        )
        candidate_data = json.loads(result)
        payloads[f"{name}_payload"] = PtxKernel.model_validate(
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
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")
    payloads = generate_kernel_payloads(Apertus, arguments)
    optimized = Apertus.from_custom_ptx(**payloads)
    _, optimized_tokens_per_second = measure(
        optimized,
        MESSAGES,
        max_new_tokens=arguments.max_new_tokens,
    )
    baseline = Apertus()
    _, baseline_tokens_per_second = measure(
        baseline,
        MESSAGES,
        max_new_tokens=arguments.max_new_tokens,
    )
    del baseline
    torch.cuda.empty_cache()
    speedup = optimized_tokens_per_second / baseline_tokens_per_second
    print(f"Baseline: {baseline_tokens_per_second:.2f} tokens/s")
    print(f"Injected PTX: {optimized_tokens_per_second:.2f} tokens/s")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
