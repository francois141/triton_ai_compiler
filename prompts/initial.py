from __future__ import annotations

from triton_ptx.helpers.kernels import extract_specification_from_operator

from .blocks import (
    commenting_rules,
    constexpr_values_block,
    correctness_rules,
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
    "float16_gemm_research",
    "triton_kernel",
    "output_contract",
)

IMPROVEMENT_CONTEXT_SECTION_NAMES = (
    "constexpr_values",
    "launch_configuration",
    "shape_information",
    "performance_rules",
    "float16_gemm_research",
    "triton_kernel",
)


def _is_float16_operator(operator):
    return type(operator).__module__.startswith("triton_ptx.kernels.level2_float16")


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
        "float16_gemm_research": (
            float16_gemm_research_rules() if include_float16_gemm_research else ""
        ),
        "triton_kernel": triton_kernel_block(spec.source),
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
):
    prompt_sections = build_prompt_sections(
        spec,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=ptx_signature,
        include_float16_gemm_research=include_float16_gemm_research,
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
    )
