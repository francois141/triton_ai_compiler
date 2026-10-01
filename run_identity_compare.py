import argparse
import logging
import re

import torch
from ptx_gym.evaluation.performance import evaluate_ptx_performance
from ptx_gym.evaluation.verification import OutputVerifier
from ptx_gym.helpers.triton import dump_kernel_ptx
from ptx_gym.kernels import resolve_kernel

LOGGER = logging.getLogger(__name__)

REQNTID_PATTERN = re.compile(
    r"^\s*\.reqntid\s+(\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?\s*$",
    re.MULTILINE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a kernel's PTX and compare it with its Triton baseline."
    )
    parser.add_argument(
        "kernel",
        help="Kernel class name, such as SoftmaxFloat16Kernel.",
    )
    return parser.parse_args()


def ptx_launch_dimensions(ptx, num_warps):
    required_threads = REQNTID_PATTERN.search(ptx)
    if required_threads is None:
        return num_warps * 32, 1, 1

    threads_x, threads_y, threads_z = required_threads.groups()
    return (
        int(threads_x),
        int(threads_y) if threads_y is not None else 1,
        int(threads_z) if threads_z is not None else 1,
    )


def main():
    args = parse_args()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to compare Triton and PTX kernels.")

        kernel_class = resolve_kernel(args.kernel)
        baseline = kernel_class()
        inputs = baseline.get_random_input()
        generated_ptx = dump_kernel_ptx(baseline, inputs)
        if generated_ptx is None:
            raise RuntimeError(f"Triton did not produce PTX for {args.kernel}.")

        threads_x, threads_y, threads_z = ptx_launch_dimensions(
            generated_ptx, baseline.num_warps
        )
        generated = kernel_class(
            ptx={
                "ptx": generated_ptx,
                "num_threads_x": threads_x,
                "num_threads_y": threads_y,
                "num_threads_z": threads_z,
                "tuning_config": dict(baseline.best_config),
            }
        )
        verifier = OutputVerifier()
        if not verifier.verify(generated):
            LOGGER.error(
                "Generated PTX output does not match Triton: %s",
                verifier.last_report,
            )
            return 1
        performance = evaluate_ptx_performance(generated, inputs)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        LOGGER.error("Failed to compare %s: %s", args.kernel, exc)
        return 1

    baseline_timing = performance["triton"]
    generated_timing = performance["ptx"]
    speedup = performance["ptx_speedup"]
    print(f"{args.kernel}: generated PTX output matches Triton")
    print(f"Baseline Triton latency (p50): {baseline_timing.p50:.4f} ms")
    print(f"Generated PTX latency (p50): {generated_timing.p50:.4f} ms")
    if speedup is None:
        print("Generated PTX speedup vs baseline: unavailable")
    else:
        print(f"Generated PTX speedup vs baseline: {speedup:.4f}x")
    return 0


if __name__ == "__main__":
    main()
