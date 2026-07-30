"""Run Apertus using PTX extracted from its base Triton kernels."""

import torch

from triton_ptx import dump_kernel_ptx
from triton_ptx.LLMs.apertus import Apertus
from triton_ptx.LLMs.apertus.kernels import (
    ApertusRMSNormKernel,
    CausalAttentionKernel,
    LinearKernel,
    RoPEKernel,
    XIELUKernel,
)

KERNEL_CLASSES = {
    "causal_attention": CausalAttentionKernel,
    "linear": LinearKernel,
    "rms_norm": ApertusRMSNormKernel,
    "rope": RoPEKernel,
    "xielu": XIELUKernel,
}


def extract_base_ptx_payloads():
    """Compile Apertus's base Triton kernels into injectable PTX payloads."""
    device = "cuda"
    dtype = torch.bfloat16
    sample_inputs = {
        "causal_attention": tuple(
            torch.rand((1, 1, 1, 128), device=device, dtype=dtype)
            for _ in range(3)
        ),
        "linear": (
            torch.rand((1, 4096), device=device, dtype=dtype),
            torch.rand((4096, 4096), device=device, dtype=dtype),
        ),
        "rms_norm": (
            torch.rand((1, 4096), device=device, dtype=dtype),
            torch.rand(4096, device=device, dtype=dtype),
            1e-5,
        ),
        "rope": (
            torch.rand((1, 1, 1, 128), device=device, dtype=dtype),
            torch.rand((1, 128), device=device, dtype=dtype),
            torch.rand((1, 128), device=device, dtype=dtype),
        ),
        "xielu": tuple(
            torch.rand(256 if index == 0 else 1, device=device, dtype=dtype)
            for index in range(5)
        ),
    }
    payloads = {}
    for name, kernel_class in KERNEL_CLASSES.items():
        kernel = kernel_class()
        ptx = dump_kernel_ptx(kernel, sample_inputs[name])
        if not ptx:
            raise RuntimeError(
                f"Triton did not produce PTX for {kernel_class.__name__}."
            )
        payloads[name] = {
            "ptx": ptx,
            "num_threads_x": kernel.num_warps * 32,
        }
    return payloads


def main():
    payloads = extract_base_ptx_payloads()
    llm = Apertus.from_custom_ptx(**payloads)
    response = llm.generate(
        [
            {
                "role": "system",
                "content": "You are a concise and helpful assistant.",
            },
            {
                "role": "user",
                "content": (
                    "You are a good Swiss citizen. On est d'accord le fromage "
                    "français est bizarre?"
                ),
            },
        ],
        max_new_tokens=640,
    )
    print(response)


if __name__ == "__main__":
    main()
