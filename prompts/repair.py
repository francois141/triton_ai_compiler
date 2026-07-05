from __future__ import annotations

import orjson

from .skills import async_load_store_skill
from triton_ptx.evaluation import EvaluatedCandidate
from triton_ptx.helpers.kernels import extract_specification_from_operator
from triton_ptx.kernels.base import TritonPTXKernel
from .blocks import (
    commenting_rules,
    correctness_rules,
    constexpr_values_block,
    launch_configuration_block,
    output_contract,
    signature_template,
    triton_kernel_block,
)
from .next import candidate_results_block
from .skills import common_ptxas_issues_skill


def repair_task(retry_index: int, max_retries: int) -> str:
    return """
# PTX Isolated Candidate Repair

You are given one failed PTX candidate for a Triton kernel and you have to repair it. 
""".strip()


def repair_rules() -> str:
    return """
## Repair Rules

- Fix only the first concrete compiler error class and make the smallest change needed.
- Preserve the exact PTX header, kernel entry name, runtime argument order, and launch metadata keys.
- Return the ptx code with the same json format, not a patch or explanation.

# The candidate intentionally uses cp.async and asynchronous shared-memory staging.

Your job is NOT to redesign the kernel.

Only fix the PTXAS compilation errors while preserving:
- cp.async
- async commit/wait groups
- shared-memory pipeline
- buffering strategy
- tiling strategy
""".strip()


def sanitizer_diagnostics_block(candidate: EvaluatedCandidate) -> str:
    result = (
        orjson.dumps(
            candidate.sanitizer_report,
            default=str,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        ).decode()
        if candidate.sanitizer_report
        else "None"
    )
    return f"""
## Compute Sanitizer Diagnostics

```json
{result}
```
""".strip()


def prompt_builder(
    spec,
    failed_candidate: EvaluatedCandidate,
    *,
    retry_index: int,
    max_retries: int,
    version: str,
    target: str,
    address_size: int,
    ptx_signature=None,
) -> str:
    sections = [
        repair_task(retry_index, max_retries),
        common_ptxas_issues_skill(),
        async_load_store_skill(),
        constexpr_values_block(spec),
        launch_configuration_block(),
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
        triton_kernel_block(spec.source),
        candidate_results_block([failed_candidate]),
        sanitizer_diagnostics_block(failed_candidate),
        repair_rules(),
        output_contract(spec),
    ]
    return "\n\n".join(sections)


def build_repair_prompt_for_operator(
    failed_candidate: EvaluatedCandidate,
    operator_cls: type[TritonPTXKernel],
    *,
    retry_index: int,
    max_retries: int,
    version: str,
    target: str,
    address_size: int,
    ptx_signature=None,
) -> str:
    spec = extract_specification_from_operator(operator_cls)
    return prompt_builder(
        spec,
        failed_candidate,
        retry_index=retry_index,
        max_retries=max_retries,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=ptx_signature,
    )
