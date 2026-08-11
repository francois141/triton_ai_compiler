from __future__ import annotations

import json

from triton_ptx import Payload

from .response_format import PtxKernel


def json_default(value):
    if hasattr(value, "detach") and hasattr(value, "numel"):
        tensor = value.detach()
        shape = list(tensor.shape)
        summary = {
            "type": value.__class__.__name__,
            "shape": shape,
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
        if tensor.numel() <= 16:
            summary["values"] = tensor.cpu().tolist()
        return summary

    if hasattr(value, "tolist"):
        return value.tolist()

    if isinstance(value, set):
        return sorted(value)

    return str(value)


def candidate_from_evaluation(evaluation):
    payload = Payload.from_input(evaluation.payload).to_launch_dict()
    return PtxKernel.model_validate(payload)


def evaluation_summary(evaluation, *, include_ptx, include_ncu=True):
    payload = candidate_from_evaluation(evaluation).model_dump(exclude_none=False)
    ptx = payload.pop("ptx", "")
    lines = [
        "## Evaluation Result",
        f"- Compiles: {evaluation.compiles}",
        f"- Correct: {evaluation.correct}",
        f"- P50: {evaluation.p50}",
        f"- P95: {evaluation.p95}",
        f"- Speedup vs. Triton: {evaluation.speedup_vs_triton}",
    ]
    message = (evaluation.message or "").strip()
    if message:
        lines.append(f"- Message: {message}")

    lines.extend(
        [
            "",
            "## Candidate Launch Metadata",
            "```json",
            json.dumps(payload, indent=2, default=json_default),
            "```",
        ]
    )
    if include_ptx:
        lines.extend(["", "## Candidate PTX", "```ptx", ptx, "```"])

    if evaluation.verifier_report:
        lines.extend(
            [
                "",
                "## Verifier Report",
                "```json",
                json.dumps(evaluation.verifier_report, indent=2, default=json_default),
                "```",
            ]
        )
    if include_ncu and evaluation.ncu_report:
        ncu_report = evaluation.ncu_report
        ncu_status = {
            key: ncu_report.get(key)
            for key in (
                "available",
                "return_code",
                "error",
            )
            if ncu_report.get(key) not in (None, "")
        }
        lines.extend(
            [
                "",
                "## Nsight Compute Status",
                "```json",
                json.dumps(ncu_status, indent=2, default=json_default),
                "```",
            ]
        )
    if evaluation.sanitizer_report:
        lines.extend(
            [
                "",
                "## Sanitizer Report",
                "```json",
                json.dumps(evaluation.sanitizer_report, indent=2, default=json_default),
                "```",
            ]
        )
    if evaluation.compile_error:
        lines.extend(
            [
                "",
                "## Compile Error",
                "```text",
                evaluation.compile_error[-2000:],
                "```",
            ]
        )
    if evaluation.timing_error:
        lines.extend(
            [
                "",
                "## Timing Error",
                "```text",
                evaluation.timing_error[-2000:],
                "```",
            ]
        )

    return "\n".join(lines)
