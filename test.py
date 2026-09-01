from __future__ import annotations

import torch
from triton_ptx.evaluation.performance import benchmark
from triton_ptx.helpers.triton import dump_kernel_ptx
from triton_ptx.kernels.level2_float16.convolution_2d import (
    Convolution2DFloat16Kernel,
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark the convolution kernel.")

    triton_kernel = Convolution2DFloat16Kernel(autotune=False)
    inputs = triton_kernel.get_random_input()
    generated_ptx = dump_kernel_ptx(triton_kernel, inputs)
    if generated_ptx is None:
        raise RuntimeError("Triton did not produce PTX for the convolution kernel.")

    ptx_kernel = Convolution2DFloat16Kernel(
        ptx={
            "ptx": generated_ptx,
            "num_threads_x": triton_kernel.num_warps * 32,
        }
    )

    triton_kernel.forward_triton(inputs)
    ptx_kernel.forward_triton(inputs, ptx=True)
    torch.cuda.synchronize()

    triton_timing = benchmark(lambda: triton_kernel.forward_triton(inputs))
    ptx_timing = benchmark(lambda: ptx_kernel.forward_triton(inputs, ptx=True))
    speedup = triton_timing.p50 / ptx_timing.p50

    print(f"Triton p50: {triton_timing.p50:.6f} ms")
    print(f"Extracted PTX p50: {ptx_timing.p50:.6f} ms")
    print(f"Extracted PTX speedup vs Triton: {speedup:.4f}x")


if __name__ == "__main__":
    main()
