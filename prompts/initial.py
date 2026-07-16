from __future__ import annotations

from .blocks import (
    commenting_rules,
    correctness_rules,
    constexpr_values_block,
    initial_task,
    launch_configuration_block,
    output_contract,
    performance_rules,
    signature_template,
    triton_kernel_block,
)
from .skills import common_ptxas_issues_skill


def build_prompt_for_kernel(data):
    system = data["system"]
    sections = [
        common_ptxas_issues_skill(),
        initial_task().strip(),
        constexpr_values_block(data),
        launch_configuration_block(),
        signature_template(
            data["parameters"],
            version=system["version"],
            target=system["target"],
            address_size=system["address_size"],
            kernel_name=data["kernel_name"],
            ptx_signature=data["ptx_signature"],
        ),
        correctness_rules(),
        commenting_rules(),
        performance_rules(),
        triton_kernel_block(data["source"]),
        output_contract(data),
    ]
    return "\n\n".join(sections)
