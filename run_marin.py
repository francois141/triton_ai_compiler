import argparse
import re
from time import perf_counter

import torch
from triton_ptx.LLMs.marin import Marin

from triton_ptx import dump_kernel_ptx

PROMPT = "The Marin wind is"


@torch.inference_mode()
def compile_ptx_payloads():
    payloads = {}
    for name, kernel_class in Marin.get_kernel_classes().items():
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
    for name, kernel_class in Marin.get_kernel_classes().items():
        inputs = kernel_class().get_random_input(fixed=True)
        injected_kernel = kernel_class(ptx=payloads[f"{name}_payload"])
        expected = injected_kernel.forward_torch(inputs)
        actual, _ = injected_kernel.forward_triton(inputs, ptx=True)
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)


def validate_response(prompt, continuation):
    words = re.findall(r"[A-Za-z]+", continuation)
    unique_word_ratio = len({word.lower() for word in words}) / max(len(words), 1)
    if len(words) < 5 or unique_word_ratio < 0.2:
        raise RuntimeError(
            f"Marin produced an incoherent continuation for {prompt!r}: "
            f"{continuation!r}"
        )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")
    torch.manual_seed(arguments.seed)

    payloads = compile_ptx_payloads()
    verify_injected_ptx(payloads)
    print(f"Verified {len(payloads)} compiled and re-injected PTX kernels.")

    llm = Marin.from_custom_ptx(**payloads)
    input_ids = llm.tokenize_messages([arguments.prompt])
    torch.cuda.synchronize(input_ids.device)
    start_time = perf_counter()
    output_ids = llm.generate_tokens(
        input_ids,
        max_new_tokens=arguments.max_new_tokens,
    )
    torch.cuda.synchronize(input_ids.device)
    elapsed_seconds = perf_counter() - start_time
    continuation = llm.decode_response(output_ids, input_ids.shape[-1])
    validate_response(arguments.prompt, continuation)
    generated_tokens = output_ids.shape[-1] - input_ids.shape[-1]

    print(llm.decode_full_response(output_ids))
    print(f"Throughput: {generated_tokens / elapsed_seconds:.2f} tokens/s")


if __name__ == "__main__":
    main()
