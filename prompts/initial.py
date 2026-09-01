from __future__ import annotations

from triton_ptx.helpers.kernels import extract_specification_from_operator

from .blocks import (
    commenting_rules,
    constexpr_values_block,
    convolution_2d_float16_rules,
    correctness_rules,
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

IMPROVEMENT_CONTEXT_SECTION_NAMES = (
    "constexpr_values",
    "launch_configuration",
    "shape_information",
    "performance_rules",
    "flash_attention",
    "float16_gemm_research",
    "convolution_memory_layout",
    "triton_kernel",
)


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
):
    return {
        "initial_task": initial_task().strip(),
        "constexpr_values": constexpr_values_block(spec),
        "launch_configuration": launch_configuration_block(spec.num_warps),
        "ptx_entry_template": signature_template(
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
            float16_gemm_research_rules() if include_float16_gemm_research else ""
        ),
        "convolution_memory_layout": (
            convolution_2d_float16_rules() if include_convolution_memory_layout else ""
        ),
        "triton_kernel": triton_kernel_block(spec.source, spec.supporting_source),
        "output_contract": output_contract(spec),
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
    )
    return render_prompt_sections(prompt_sections, INITIAL_PROMPT_SECTION_NAMES)


def build_prompt_for_operator(
    operator,
    *,
    version,
    target,
    address_size,
    ptx_signature=None,
):
    spec = extract_specification_from_operator(operator)
    return prompt_builder(
        spec,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=ptx_signature,
        include_float16_gemm_research=_is_float16_operator(operator),
        include_flash_attention=_is_flash_attention_float16_operator(operator),
        include_convolution_memory_layout=_is_convolution_2d_float16_operator(operator),
    )


def build_prompt_sections_for_operator(
    operator,
    *,
    version,
    target,
    address_size,
    ptx_signature=None,
):
    spec = extract_specification_from_operator(operator)
    return build_prompt_sections(
        spec,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=ptx_signature,
        include_float16_gemm_research=_is_float16_operator(operator),
        include_flash_attention=_is_flash_attention_float16_operator(operator),
        include_convolution_memory_layout=_is_convolution_2d_float16_operator(operator),
    )
