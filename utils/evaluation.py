
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


def evaluation_summary(evaluation, *, include_ptx):
    payload = candidate_from_evaluation(evaluation).model_dump(exclude_none=False)
    if not include_ptx:
        payload["ptx"] = "<omitted>"

    fields = {
        "compiles": evaluation.compiles,
        "correct": evaluation.correct,
        "p50": evaluation.p50,
        "p95": evaluation.p95,
        "speedup_vs_triton": evaluation.speedup_vs_triton,
        "message": (evaluation.message or "").strip(),
        "candidate": payload,
    }
    if evaluation.verifier_report:
        fields["verifier_report"] = evaluation.verifier_report
    if evaluation.ncu_report:
        fields["ncu_report"] = json.dumps(
            evaluation.ncu_report,
            default=json_default,
        )[-4000:]
    if evaluation.compile_error:
        fields["compile_error"] = evaluation.compile_error[-2000:]
    if evaluation.timing_error:
        fields["timing_error"] = evaluation.timing_error[-2000:]

    return json.dumps(fields, indent=2, default=json_default)
