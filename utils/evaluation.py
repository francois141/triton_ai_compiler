from __future__ import annotations

import json
import weakref

_CANDIDATES_BY_EVALUATION_ID = {}


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
        return normalize_nested_json(summary)

    if hasattr(value, "tolist"):
        return normalize_nested_json(value.tolist())

    if isinstance(value, set):
        return normalize_nested_json(sorted(value))

    return normalize_nested_json(str(value))


def normalize_nested_json(value):
    if isinstance(value, dict):
        return {key: normalize_nested_json(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [normalize_nested_json(item) for item in value]

    if not isinstance(value, str):
        return value

    try:
        decoded_value = json.loads(value)
    except json.JSONDecodeError:
        return value

    return (
        normalize_nested_json(decoded_value)
        if isinstance(decoded_value, (dict, list))
        else value
    )


def register_evaluated_candidate(evaluation, candidate):
    evaluation_id = id(evaluation)
    _CANDIDATES_BY_EVALUATION_ID[evaluation_id] = candidate
    weakref.finalize(evaluation, _CANDIDATES_BY_EVALUATION_ID.pop, evaluation_id, None)


def candidate_from_evaluation(evaluation):
    try:
        return _CANDIDATES_BY_EVALUATION_ID[id(evaluation)]
    except KeyError as error:
        raise ValueError("The evaluated candidate is unavailable.") from error


def _fenced_section(title, language, body):
    return ["", f"## {title}", f"```{language}", body, "```"]


def _json_section(title, value):
    return _fenced_section(
        title,
        "json",
        json.dumps(normalize_nested_json(value), indent=2, default=json_default),
    )


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

    lines.extend(_json_section("Candidate Launch Metadata", payload))
    if include_ptx:
        lines.extend(_fenced_section("Candidate PTX", "ptx", ptx))
    if evaluation.verifier_report:
        lines.extend(_json_section("Verifier Report", evaluation.verifier_report))
    if include_ncu and evaluation.ncu_report:
        ncu_report = evaluation.ncu_report
        ncu_status = {
            key: ncu_report.get(key)
            for key in ("available", "return_code", "error")
            if ncu_report.get(key) not in (None, "")
        }
        lines.extend(_json_section("Nsight Compute Status", ncu_status))
    if evaluation.sanitizer_report:
        lines.extend(_json_section("Sanitizer Report", evaluation.sanitizer_report))
    if evaluation.compile_error:
        lines.extend(
            _fenced_section("Compile Error", "text", evaluation.compile_error[-2000:])
        )
    if evaluation.timing_error:
        lines.extend(
            _fenced_section("Timing Error", "text", evaluation.timing_error[-2000:])
        )

    return "\n".join(lines)
