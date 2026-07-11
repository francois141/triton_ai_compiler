from __future__ import annotations

import math

from triton_ptx.evaluation import EvaluatedCandidate
from triton_ptx.helpers.kernels import (
    PTX_LAUNCH_KEYS,
    extract_specification_from_operator,
)
from triton_ptx.kernels.base import TritonPTXKernel
from .blocks import (
    commenting_rules,
    correctness_rules,
    constexpr_values_block,
    follow_up_task,
    launch_configuration_block,
    output_contract,
    performance_rules,
    signature_template,
    triton_kernel_block,
)
from .skills import async_load_store_skill, common_ptxas_issues_skill


def _format_metric(value):
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def _format_extra_payload_keys(candidate):
    extra_payload_keys = {
        key: value
        for key, value in candidate.payload.items()
        if key != "ptx" and key not in PTX_LAUNCH_KEYS
    }
    if not extra_payload_keys:
        return "None"

    return "\n".join(
        f"- {name}: {value!r}" for name, value in sorted(extra_payload_keys.items())
    )


def _format_launch_metadata(candidate):
    launch_metadata = {
        key: candidate.payload[key]
        for key in sorted(PTX_LAUNCH_KEYS)
        if isinstance(candidate.payload, dict) and key in candidate.payload
    }
    if not launch_metadata:
        return "None"

    return "\n".join(f"- {name}: {value!r}" for name, value in launch_metadata.items())


def _candidate_block(candidate, display_index):
    ptx_code = str(candidate.payload.get("ptx", "")).strip() or "<missing PTX payload>"
    diagnostics_block = ""
    if not candidate.correct:
        diagnostics_block = f"""

Compiler stdout:
```text
{candidate.compile_output.strip() or "None"}
```

Compiler stderr:
```text
{candidate.compile_error.strip() or "None"}
```

Timing error:
```text
{candidate.timing_error.strip() or "None"}
```
"""

    return f"""
## Candidate {display_index}

- Compiles: {"yes" if candidate.compiles else "no"}
- Correct: {"yes" if candidate.correct else "no"}
- Passed: {"yes" if candidate.passed else "no"}
- Message: {candidate.message.strip() or "None"}
- p20: {_format_metric(candidate.p20)}
- p50: {_format_metric(candidate.p50)}
- p80: {_format_metric(candidate.p80)}
- p90: {_format_metric(candidate.p90)}
- p95: {_format_metric(candidate.p95)}
- p99: {_format_metric(candidate.p99)}

Extra PTX payload keys:
{_format_extra_payload_keys(candidate)}

Launch metadata:
{_format_launch_metadata(candidate)}
{diagnostics_block}

PTX:
```ptx
{ptx_code}
```
""".strip()


def candidate_results_block(candidates):
    if not candidates:
        return "\n\n".join(
            [
                "## Candidate Results",
                "No evaluated candidates were provided. Generate strong answers from the Triton kernel specification alone.",
            ]
        )

    sections = [
        "## Candidate Results",
        "Use these results to keep strong ideas from successful candidates and avoid repeating choices that caused compilation, correctness, or performance failures.",
    ]
    sections.extend(
        _candidate_block(candidate, display_index)
        for display_index, candidate in enumerate(candidates, start=1)
    )
    return "\n\n".join(sections)


def follow_up_rules():
    return """
## Follow-Up Rules

- Study the candidate PTX, compiler feedback, correctness outcome, and runtime percentiles before producing new answers.
- Prefer ideas from candidates that compiled, passed correctness, and achieved lower p50 runtime.
- If a candidate failed, infer the likely cause from the PTX and feedback and avoid repeating that mistake.
- You may combine strong ideas from multiple candidates into a better implementation.
- Improve performance without sacrificing correctness or signature compatibility.
- Return distinct answers, not trivial rewrites of the same PTX.
""".strip()


def prompt_builder(
    spec,
    candidates,
    *,
    version,
    target,
    address_size,
    num_answers = 5,
    ptx_signature=None,
):
    sections = [
        async_load_store_skill(),
        common_ptxas_issues_skill(),
        follow_up_task().strip(),
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
        performance_rules(target, version, spec),
        triton_kernel_block(spec.source),
        candidate_results_block(candidates),
        follow_up_rules(),
        output_contract(spec),
    ]
    return "\n\n".join(sections)


def build_follow_up_prompt_for_operator(
    candidates,
    operator_cls,
    *,
    version,
    target,
    address_size,
    num_answers = 5,
    ptx_signature=None,
):
    spec = extract_specification_from_operator(operator_cls)
    return prompt_builder(
        spec,
        candidates or [],
        version=version,
        target=target,
        address_size=address_size,
        num_answers=num_answers,
        ptx_signature=ptx_signature,
    )
