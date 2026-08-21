#!/usr/bin/env python3

import torch
from triton_ptx.helpers.triton import dump_kernel_ptx
from triton_ptx.kernels import SoftmaxKernel

from triton_ptx import Payload, TritonPTXCandidateEvaluator

WARMUP_ITERATIONS = 20
MEASUREMENT_ITERATIONS = 100


def average_triton_latency_ms(kernel: SoftmaxKernel) -> float:
    """Measure the average launch latency of the baseline Triton kernel."""
    inputs = kernel.get_random_input()
    for _ in range(WARMUP_ITERATIONS):
        print("coucou")
        kernel.forward_triton(inputs)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(MEASUREMENT_ITERATIONS):
        kernel.forward_triton(inputs)
    end_event.record()
    end_event.synchronize()
    return start_event.elapsed_time(end_event) / MEASUREMENT_ITERATIONS


def main():
    try:
        kernel = SoftmaxKernel()
        ptx = dump_kernel_ptx(kernel)
        if not ptx:
            raise RuntimeError("Triton did not produce PTX for SoftmaxKernel.")

        print("evaluation")

        report = TritonPTXCandidateEvaluator(SoftmaxKernel, operator=kernel).evaluate(
            Payload.from_input(
                {
                    "ptx": ptx,
                    "threads_x": kernel.num_warps * 32,
                }
            )
        )

        print("end evaluataion")
    except (OSError, RuntimeError, TypeError, ValueError):
        print("Failure")
        return 1

    if not report.passed:
        print("Failure")
        return 1

    try:
        latency_ms = average_triton_latency_ms(kernel)
    except RuntimeError:
        print("Failure")
        return 1

    print(f"Average Triton kernel latency: {latency_ms:.4f} ms")
    print("Success")
    return 0


if __name__ == "__main__":
    main()
