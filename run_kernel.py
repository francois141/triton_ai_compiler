#!/usr/bin/env python3

import argparse
import logging

import torch
from ptx_gym.evaluation.performance import benchmark
from ptx_gym.evaluation.verification import OutputVerifier
from ptx_gym.kernels import resolve_kernel

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify and benchmark a baseline Triton kernel."
    )
    parser.add_argument(
        "kernel",
        help="Kernel class name, such as SoftmaxFloat16Kernel.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to run a Triton kernel.")
        kernel = resolve_kernel(args.kernel)()
        tolerance = getattr(kernel, "verification_tolerance", 1e-3)
        verifier = OutputVerifier(rtol=tolerance, atol=tolerance)
        if not verifier.verify_triton_vs_torch(kernel):
            LOGGER.error("Triton output does not match Torch: %s", verifier.last_report)
            return 1

        inputs = kernel.get_random_input()
        timing = benchmark(lambda: kernel.forward_triton(inputs))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        LOGGER.error("Failed to run %s: %s", args.kernel, exc)
        return 1

    print(f"{args.kernel}: Triton output matches Torch")
    print(f"Triton latency (p50): {timing.p50:.4f} ms")
    return 0


if __name__ == "__main__":
    main()
