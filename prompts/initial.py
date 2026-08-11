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
def prompt_builder(
    spec,
    *,
    version,
    target,
    address_size,
    ptx_signature=None,
    include_float16_gemm_research=False,
):
    sections = [
        initial_task().strip(),
        constexpr_values_block(spec),
        launch_configuration_block(spec.num_warps),
        signature_template(
            spec.parameters,
            version=version,
            target=target,
            address_size=address_size,
            kernel_name=spec.kernel_name,
            ptx_signature=ptx_signature,
        ),
        shape_information_block(spec.shape_information),
        correctness_rules(spec.num_warps),
        commenting_rules(),
        performance_rules(spec.num_warps),
        triton_kernel_block(spec.source),
        output_contract(spec),
    ]
    if include_float16_gemm_research:
        sections.insert(-2, float16_gemm_research_rules())
    return "\n\n".join(sections)


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
        include_float16_gemm_research=(
            type(operator).__name__ == "MatrixMultiplicationFloat16"
        ),
    )
