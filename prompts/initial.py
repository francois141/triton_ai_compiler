from __future__ import annotations

from triton_ptx.helpers.kernels import extract_specification_from_operator
from triton_ptx.helpers.triton import get_kernel_shared_memory_bytes

from .blocks import (
    anthropic_float16_gemm_research_rules,
    anthropic_initial_task,
    anthropic_output_contract,
    anthropic_signature_template,
    commenting_rules,
    constexpr_values_block,
    convolution_2d_float16_rules,
    correctness_rules,
    dynamic_shared_memory_allocation,
    flash_attention_float16_rules,
    float16_gemm_research_rules,
    initial_task,
    launch_configuration_block,
    output_contract,
    performance_rules,
    shape_information_block,
    signature_template,
    triton_kernel_block,
)

INITIAL_PROMPT_SECTION_NAMES = (
    "initial_task",
    "constexpr_values",
    "launch_configuration",
    "dynamic_shared_memory",
    "ptx_entry_template",
    "shape_information",
    "correctness_rules",
    "commenting_rules",
    "performance_rules",
    "flash_attention",
    "float16_gemm_research",
    "convolution_memory_layout",
    "triton_kernel",
    "output_contract",
)

ANTHROPIC_INITIAL_PROMPT_SECTION_NAMES = (
    "initial_task",
    "constexpr_values",
    "launch_configuration",
    "dynamic_shared_memory",
    "ptx_entry_template",
    "shape_information",
    "correctness_rules",
    "triton_kernel",
    "output_contract",
)

IMPROVEMENT_CONTEXT_SECTION_NAMES = (
    "constexpr_values",
    "launch_configuration",
    "dynamic_shared_memory",
    "shape_information",
    "performance_rules",
    "flash_attention",
    "float16_gemm_research",
    "convolution_memory_layout",
    "triton_kernel",
)


def initial_prompt_section_names(provider):
    if provider == "anthropic":
        return ANTHROPIC_INITIAL_PROMPT_SECTION_NAMES
    if provider in {"openai", "openrouter"}:
        return INITIAL_PROMPT_SECTION_NAMES
    raise ValueError(f"Unsupported provider: {provider}")


def _is_float16_operator(operator):
    return _operator_class(operator).__module__.startswith(
        "triton_ptx.kernels.level2_float16"
    )


def _operator_class(operator):
    return operator if isinstance(operator, type) else type(operator)


def _is_convolution_2d_float16_operator(operator):
    return _operator_class(operator).__name__ == "Convolution2DFloat16Kernel"


def _is_flash_attention_float16_operator(operator):
    return _operator_class(operator).__name__ == "FlashAttentionFloat16Kernel"


def render_prompt_sections(prompt_sections, section_names):
    return "\n\n".join(
        prompt_sections[section_name]
        for section_name in section_names
        if prompt_sections.get(section_name)
    )


def build_prompt_sections(
    spec,
    *,
    version,
    target,
    address_size,
    ptx_signature=None,
    include_float16_gemm_research=False,
    include_flash_attention=False,
    include_convolution_memory_layout=False,
    provider="openai",
    shared_memory_bytes=None,
):
    if provider not in {"anthropic", "openai", "openrouter"}:
        raise ValueError(f"Unsupported provider: {provider}")
    is_anthropic = provider == "anthropic"
    return {
        "initial_task": (
            anthropic_initial_task().strip() if is_anthropic else initial_task().strip()
        ),
        "constexpr_values": constexpr_values_block(spec),
        "launch_configuration": launch_configuration_block(spec.num_warps),
        "dynamic_shared_memory": dynamic_shared_memory_allocation(shared_memory_bytes),
        "ptx_entry_template": (
            anthropic_signature_template if is_anthropic else signature_template
        )(
            spec.parameters,
            version=version,
            target=target,
            address_size=address_size,
            kernel_name=spec.kernel_name,
            ptx_signature=ptx_signature,
        ),
        "shape_information": shape_information_block(spec.shape_information),
        "correctness_rules": correctness_rules(spec.num_warps),
        "commenting_rules": commenting_rules(),
        "performance_rules": performance_rules(spec.num_warps),
        "flash_attention": (
            flash_attention_float16_rules() if include_flash_attention else ""
        ),
        "float16_gemm_research": (
            (
                anthropic_float16_gemm_research_rules(
                    shared_memory_bytes=shared_memory_bytes,
                )
                if is_anthropic
                else float16_gemm_research_rules()
            )
            if include_float16_gemm_research
            else ""
        ),
        "convolution_memory_layout": (
            convolution_2d_float16_rules() if include_convolution_memory_layout else ""
        ),
        "triton_kernel": triton_kernel_block(spec.source, spec.supporting_source),
        "output_contract": (
            anthropic_output_contract(spec) if is_anthropic else output_contract(spec)
        ),
    }


def prompt_builder(
    spec,
    *,
    version,
    target,
    address_size,
    ptx_signature=None,
    include_float16_gemm_research=False,
    include_flash_attention=False,
    include_convolution_memory_layout=False,
    provider="openai",
    shared_memory_bytes=None,
):
    prompt_sections = build_prompt_sections(
        spec,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=ptx_signature,
        include_float16_gemm_research=include_float16_gemm_research,
        include_flash_attention=include_flash_attention,
        include_convolution_memory_layout=include_convolution_memory_layout,
        provider=provider,
        shared_memory_bytes=shared_memory_bytes,
    )
    return render_prompt_sections(
        prompt_sections, initial_prompt_section_names(provider)
    )


def build_prompt_for_operator(
    operator,
    *,
    version,
    target,
    address_size,
    ptx_signature=None,
    provider="openai",
):
    spec = extract_specification_from_operator(operator)
    include_float16_gemm_research = _is_float16_operator(operator)
    return prompt_builder(
        spec,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=ptx_signature,
        include_float16_gemm_research=include_float16_gemm_research,
        include_flash_attention=_is_flash_attention_float16_operator(operator),
        include_convolution_memory_layout=_is_convolution_2d_float16_operator(operator),
        provider=provider,
        shared_memory_bytes=get_kernel_shared_memory_bytes(operator),
    )


def build_prompt_sections_for_operator(
    operator,
    *,
    version,
    target,
    address_size,
    ptx_signature=None,
    provider="openai",
):
    spec = extract_specification_from_operator(operator)
    include_float16_gemm_research = _is_float16_operator(operator)
    return build_prompt_sections(
        spec,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=ptx_signature,
        include_float16_gemm_research=include_float16_gemm_research,
        include_flash_attention=_is_flash_attention_float16_operator(operator),
        include_convolution_memory_layout=_is_convolution_2d_float16_operator(operator),
        provider=provider,
        shared_memory_bytes=get_kernel_shared_memory_bytes(operator),
    )
