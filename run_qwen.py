import argparse
import re
from time import perf_counter

import torch
from triton_ptx.LLMs.qwen import Qwen

from triton_ptx import dump_kernel_ptx

PROMPT = "Give me a short introduction to large language models."
TOPIC_TERMS = ("ai", "language", "model", "text", "training", "data")


@torch.inference_mode()
def compile_ptx_payloads():
    payloads = {}
    for name, kernel_class in Qwen.get_kernel_classes().items():
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
def verify_injected_ptx(payloads):
    for name, kernel_class in Qwen.get_kernel_classes().items():
        inputs = kernel_class().get_random_input(fixed=True)
        injected_kernel = kernel_class(ptx=payloads[f"{name}_payload"])
        expected = injected_kernel.forward_torch(inputs)
        actual, _ = injected_kernel.forward_triton(inputs, ptx=True)
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)

    rms_inputs = (
        torch.rand((16, 128), device="cuda", dtype=torch.bfloat16),
        torch.rand(128, device="cuda", dtype=torch.bfloat16),
        1e-6,
    )
    rms_kernel_class = Qwen.get_kernel_classes()["rms_norm"]
    rms_kernel = rms_kernel_class(ptx=payloads["rms_norm_payload"])
    expected = rms_kernel.forward_torch(rms_inputs)
    actual, _ = rms_kernel.forward_triton(rms_inputs, ptx=True)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


def validate_response(thinking_content, content, prompt):
    response = f"{thinking_content} {content}".strip()
    words = re.findall(r"[A-Za-z]+", response)
    normalized_response = response.lower()
    is_invalid = len(response) < 2
    if prompt == PROMPT:
        is_invalid = len(words) < 5 or not any(
            term in normalized_response for term in TOPIC_TERMS
        )
    if is_invalid:
        raise RuntimeError(f"Qwen produced an incoherent response: {response!r}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")

    payloads = compile_ptx_payloads()
    verify_injected_ptx(payloads)
    print(f"Verified {len(payloads)} compiled and re-injected PTX kernels.")

    llm = Qwen.from_custom_ptx(
        enable_thinking=not arguments.disable_thinking,
        **payloads,
    )
    messages = [{"role": "user", "content": arguments.prompt}]
    input_ids = llm.tokenize_messages(messages)
    torch.cuda.synchronize(input_ids.device)
    start_time = perf_counter()
    output_ids = llm.generate_tokens(
        input_ids,
        max_new_tokens=arguments.max_new_tokens,
    )
    torch.cuda.synchronize(input_ids.device)
    elapsed_seconds = perf_counter() - start_time
    thinking_content, content = llm.decode_parts(output_ids, input_ids.shape[-1])
    validate_response(thinking_content, content, arguments.prompt)
    generated_tokens = output_ids.shape[-1] - input_ids.shape[-1]

    print("thinking content:", thinking_content)
    print("content:", content)
    print(f"Throughput: {generated_tokens / elapsed_seconds:.2f} tokens/s")


if __name__ == "__main__":
    main()
