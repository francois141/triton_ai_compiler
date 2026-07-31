"""Compile Apertus kernels and compare injected-PTX throughput with baseline."""

import argparse

import torch

from triton_ptx import dump_kernel_ptx
from triton_ptx.LLMs import measure
from triton_ptx.LLMs.apertus import Apertus

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


@torch.inference_mode()
def compile_kernel_payloads(llm_class):
    """Compile each declared kernel once and return injectable PTX payloads."""
    payloads = {}
    for name, kernel_class in llm_class.get_kernel_classes().items():
        kernel = kernel_class()
        ptx = dump_kernel_ptx(kernel, kernel.get_random_input())
        if not ptx:
            raise RuntimeError(
                f"Triton did not produce PTX for {kernel_class.__name__}."
            )
        payloads[name] = {
            "ptx": ptx,
            "num_threads_x": kernel.num_warps * 32,
        }
        print(f"Compiled {name}.")
    return payloads


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=100)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")

    baseline = Apertus()
    _, baseline_tokens_per_second = measure(
        baseline,
        MESSAGES,
        max_new_tokens=arguments.max_new_tokens,
    )
    del baseline
    torch.cuda.empty_cache()
    payloads = compile_kernel_payloads(Apertus)
    optimized = Apertus.from_custom_ptx(**payloads)
    _, optimized_tokens_per_second = measure(
        optimized,
        MESSAGES,
        max_new_tokens=arguments.max_new_tokens,
    )
    speedup = optimized_tokens_per_second / baseline_tokens_per_second
    print(f"Baseline: {baseline_tokens_per_second:.2f} tokens/s")
    print(f"Injected PTX: {optimized_tokens_per_second:.2f} tokens/s")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
