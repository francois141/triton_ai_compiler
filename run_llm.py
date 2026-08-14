import argparse

import torch
from triton_ptx.LLMs import measure
from triton_ptx.LLMs.apertus import ApertusLLM
from triton_ptx.LLMs.gemma import GemmaLLM
from triton_ptx.LLMs.marin import MarinLLM
from triton_ptx.LLMs.qwen import QwenLLM

from triton_ptx import dump_kernel_ptx

LLM_CLASSES = {
    "apertus": ApertusLLM,
    "gemma": GemmaLLM,
    "marin": MarinLLM,
    "qwen": QwenLLM,
}
DEFAULT_PROMPT = "What is the capital of France?"


@torch.inference_mode()
def extract_ptx_payloads(llm_class):
    payloads = {}
    for name, kernel_class in llm_class.get_kernel_classes().items():
        kernel = kernel_class()
        inputs = kernel.get_random_input(fixed=True)
        ptx = dump_kernel_ptx(kernel, inputs)
        if not ptx:
            raise RuntimeError(
                f"Triton did not produce PTX for {kernel_class.__name__}."
            )
        payloads[f"{name}_payload"] = {
            "ptx": ptx,
            "num_threads_x": kernel.num_warps * 32,
            "tuning_config": kernel.best_config,
        }
    return payloads


@torch.inference_mode()
def verify_ptx_payloads(llm_class, payloads):
    for name, kernel_class in llm_class.get_kernel_classes().items():
        kernel = kernel_class(ptx=payloads[f"{name}_payload"])
        inputs = kernel.get_random_input(fixed=True)
        expected = kernel.forward_torch(inputs)
        actual, _ = kernel.forward_triton(inputs, ptx=True)
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run one Triton-backed LLM, optionally with re-injected PTX."
    )
    parser.add_argument(
        "-llm",
        "--llm",
        choices=tuple(LLM_CLASSES),
        default="qwen",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "-inject-ptx",
        "--inject-ptx",
        action="store_true",
        help="Extract each Triton kernel's current PTX and inject it into the LLM.",
    )
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def create_llm(name, payloads, disable_thinking):
    llm_class = LLM_CLASSES[name]
    arguments = dict(payloads)
    if name == "qwen":
        arguments["enable_thinking"] = not disable_thinking
    return llm_class.from_custom_ptx(**arguments)


def main():
    arguments = parse_arguments()
    if arguments.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")
    if arguments.disable_thinking and arguments.llm != "qwen":
        raise ValueError("--disable-thinking is only supported for Qwen.")
    if not torch.cuda.is_available():
        raise RuntimeError("LLM inference requires a CUDA GPU.")

    llm_class = LLM_CLASSES[arguments.llm]
    payloads = {}
    if arguments.inject_ptx:
        payloads = extract_ptx_payloads(llm_class)
        verify_ptx_payloads(llm_class, payloads)
        print(f"Verified and re-injected {len(payloads)} PTX kernels.")

    llm = create_llm(arguments.llm, payloads, arguments.disable_thinking)
    response, tokens_per_second = measure(
        llm,
        [{"role": "user", "content": arguments.prompt}],
        max_new_tokens=arguments.max_new_tokens,
    )
    print(response)
    print(f"Throughput: {tokens_per_second:.2f} tokens/s")


if __name__ == "__main__":
    main()
