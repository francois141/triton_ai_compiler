from __future__ import annotations

from triton_ptx.helpers.kernels import extract_specification_from_operator
from .blocks import (
    commenting_rules,
    correctness_rules,
    constexpr_values_block,
    extracted_signature_information,
    initial_task,
    num_warps_block,
    output_contract,
    performance_rules,
    ptx_header,
    signature_template,
    triton_kernel_block,
)
from .skills import async_load_store_skill, common_ptxas_issues_skill


# TODO: Dehardcode the target here
def prompt_builder(
    spec,
    *,
    version,
    target,
    address_size,
    num_answers=5,
    ptx_signature=None,
):
    sections = [
        async_load_store_skill(),
        common_ptxas_issues_skill(),
        initial_task().strip(),
        ptx_header()
        .format(
            version=version,
            target=target,
            address_size=address_size,
        )
        .strip(),
        extracted_signature_information(spec.parameters),
        constexpr_values_block(spec),
        num_warps_block(spec),
        signature_template(
            spec.parameters,
            version=version,
            target=target,
            address_size=address_size,
            kernel_name=spec.kernel_name,
            ptx_signature=ptx_signature,
        ),
        correctness_rules(),
        commenting_rules(),
        performance_rules(target, version, spec),
        triton_kernel_block(spec.source),
        output_contract(spec),
    ]
    return "\n\n".join(sections)


def build_prompt_for_operator(
    operator,
    *,
    version,
    target,
    address_size,
    num_answers=5,
    ptx_signature=None,
):
    spec = extract_specification_from_operator(operator)
    return prompt_builder(
        spec,
        version=version,
        target=target,
        address_size=address_size,
        num_answers=num_answers,
        ptx_signature=ptx_signature,
    )
