import pytest
import torch
from triton_ptx.LLMs.apertus import ApertusLLM
from triton_ptx.LLMs.gemma import GemmaLLM
from triton_ptx.LLMs.marin import MarinLLM
from triton_ptx.LLMs.qwen import QwenLLM

CAPITAL_PROMPT = [
    {
        "role": "user",
        "content": ("What is the capital of France? Answer in a complete sentence."),
    }
]


def create_llm(llm_class, **arguments):
    arguments.update(
        {
            f"{name}_payload": {"tuning_config": {}}
            for name in llm_class.get_kernel_classes()
        }
    )
    return llm_class.from_custom_ptx(**arguments)


def assert_next_tokens(llm, messages, expected_tokens):
    input_ids = llm.tokenize_messages(messages)
    output_ids = llm.generate_tokens(input_ids, max_new_tokens=len(expected_tokens))
    actual_tokens = output_ids[0, input_ids.shape[-1] :].tolist()
    assert actual_tokens == expected_tokens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_apertus_next_tokens():
    llm = create_llm(ApertusLLM)
    try:
        assert_next_tokens(
            llm,
            CAPITAL_PROMPT,
            [1784, 8961, 1307, 5498, 1395, 6993],
        )
    finally:
        del llm
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gemma_next_tokens():
    llm = create_llm(GemmaLLM)
    try:
        assert_next_tokens(
            llm,
            CAPITAL_PROMPT,
            [818, 5279, 529, 7001, 563, 9079],
        )
    finally:
        del llm
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_marin_next_tokens():
    llm = create_llm(MarinLLM)
    try:
        assert_next_tokens(
            llm,
            ["1, 2, 3,"],
            [220, 19, 11, 220, 20],
        )
    finally:
        del llm
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qwen_next_tokens():
    llm = create_llm(QwenLLM, enable_thinking=False)
    try:
        assert_next_tokens(
            llm,
            CAPITAL_PROMPT,
            [785, 6722, 315, 9625, 374, 12095],
        )
    finally:
        del llm
        torch.cuda.empty_cache()
