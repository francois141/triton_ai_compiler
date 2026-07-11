from __future__ import annotations

import argparse
import atexit
import json
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import NotFoundError, OpenAI

LOCAL_TRITON_PTX_ROOT = Path(__file__).resolve().parent / "triton_ptx"
if LOCAL_TRITON_PTX_ROOT.is_dir():
    sys.path.insert(0, str(LOCAL_TRITON_PTX_ROOT))

from prompts import build_prompt_for_operator  # noqa: E402
from llm_endpoint.base import PtxKernel  # noqa: E402
from triton_ptx import (  # noqa: E402
    Payload,
    TritonPTXCandidateEvaluator,
    dump_kernel_ptx,
    get_ptx_system_config,
    parse_ptx_signature,
    resolve_kernel,
)
from utils.pricing import TokenCounts, estimate_token_cost  # noqa: E402

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "SYSTEM.md"
PAYLOAD_SCHEMA = {
    "type": "object",
    "properties": {
        "ptx": {"type": "string", "minLength": 1},
        "num_threads_x": {"type": "integer", "minimum": 1},
        "num_threads_y": {
            "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]
        },
        "num_threads_z": {
            "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]
        },
    },
    "required": ["ptx", "num_threads_x", "num_threads_y", "num_threads_z"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "ptx_kernel",
    "strict": True,
    "schema": PAYLOAD_SCHEMA,
}
IMPROVEMENT_PLAN_FORMAT = {
    "type": "json_schema",
    "name": "improvement_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "improvements": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "rationale": {"type": "string", "minLength": 1},
                        "instruction": {"type": "string", "minLength": 1},
                    },
                    "required": ["name", "rationale", "instruction"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["improvements"],
        "additionalProperties": False,
    },
}
COST_LOG_PATH = Path(__file__).resolve().parent / "costs.txt"
DEFAULT_PTX_SKILL_ROOT = Path(__file__).resolve().parent / "ptx_skill"
DEFAULT_PTX_SKILL_SUBDIR = "ptx_skill"
RESPONSE_RETRY_ATTEMPTS = 6
DEFAULT_REPAIR_ATTEMPTS = 2
WEB_SEARCH_COST_PER_CALL = 10.00 / 1_000
ASYNC_LOAD_STORE_INSTRUCTION = """
## Async Load/Store Requirement

Explicitly use async loads and stores where the PTX target supports them. Use
asynchronous global-to-shared loads/staging whenever legal, and keep final
global stores coalesced with valid `st.global` instructions.
""".strip()
def _build_tools(skill_id: str | None) -> list[dict[str, Any]]:
    """Build the Responses API tools for stateless optimization calls."""
    tools: list[dict[str, Any]] = []#[{"type": "web_search"}]
    if skill_id is not None:
        tools.append(
            {
                "type": "shell",
                "environment": {
                    "type": "container_auto",
                    "skills": [
                        {
                            "type": "skill_reference",
                            "skill_id": skill_id,
                            "version": "latest",
                        }
                    ],
                },
            }
        )
    return tools


def build_initial_prompt(kernel_name: str) -> str:
    """Build the optimization prompt for a kernel.

    Args:
        kernel_name: Registered Triton PTX kernel class name.

    Returns:
        Complete initial optimization prompt.
    """
    kernel_cls = resolve_kernel(kernel_name)
    version, target, address_size = get_ptx_system_config()
    signature = parse_ptx_signature(dump_kernel_ptx(kernel_cls()))
    prompt = build_prompt_for_operator(
        kernel_cls,
        num_answers=1,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=signature,
    )
    return f"{prompt}\n\n{ASYNC_LOAD_STORE_INSTRUCTION}"


def build_continuation_prompt(kernel_name: str) -> str:
    """Build the prompt prefix used when continuing from a starting candidate.

    Args:
        kernel_name: Registered Triton PTX kernel class name.

    Returns:
        Compact continuation prompt that avoids asking for an initial candidate.
    """
    return f"""Continue optimizing the verified PTX candidate for {kernel_name}.
Do not restart from the initial kernel prompt or generate a fresh baseline.
Use the current best verified candidate as the source of truth and only propose
targeted changes that preserve the required PTX JSON response schema.

{ASYNC_LOAD_STORE_INSTRUCTION}""".strip()


def _json_default(value: object) -> object:
    """Return a JSON-safe representation for non-standard diagnostic objects."""
    if hasattr(value, "detach") and hasattr(value, "numel"):
        tensor = value.detach()
        shape = list(tensor.shape)
        summary: dict[str, object] = {
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


def _write_trace(trace_path: Path | None, events: list[dict[str, Any]]) -> None:
    if trace_path is not None:
        trace_path.write_text(
            json.dumps(events, indent=2, default=_json_default),
            encoding="utf-8",
        )


def _validate_skill_dir(skill_dir: Path) -> None:
    """Validate that a directory is an OpenAI skill package."""
    if not skill_dir.is_dir():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")
    if not (skill_dir / "SKILL.md").is_file():
        raise FileNotFoundError(f"Skill file not found: {skill_dir / 'SKILL.md'}")


def _zip_skill_dir(skill_dir: Path, zip_path: Path) -> None:
    """Create a skill ZIP preserving the package directory name."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in skill_dir.rglob("*"):
            if file_path.is_file() and ".git" not in file_path.parts:
                archive.write(file_path, file_path.relative_to(skill_dir.parent))


def _upload_skill(
    client: OpenAI,
    *,
    skill_dir: Path | None,
    skill_subdir: str,
) -> str:
    """Upload a submodule-backed or explicitly provided skill directory."""
    with tempfile.TemporaryDirectory(prefix="openai_ptx_skill_") as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        resolved_skill_dir = (
            DEFAULT_PTX_SKILL_ROOT / skill_subdir if skill_dir is None else skill_dir
        )

        _validate_skill_dir(resolved_skill_dir)
        zip_path = temp_dir / f"{resolved_skill_dir.name}.zip"
        _zip_skill_dir(resolved_skill_dir, zip_path)

        with zip_path.open("rb") as skill_file:
            skill = client.skills.create(files=[skill_file])
    return skill.id


def _get_nested_int(value: Any, *keys: str) -> int:
    for key in keys:
        if value is None:
            return 0
        if isinstance(value, dict):
            value = value.get(key)
        else:
            value = getattr(value, key, None)
    return int(value or 0)


def _estimate_response_cost(
    response: Any, model: str
) -> tuple[float | None, dict[str, int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "output_tokens": 0,
            "web_search_calls": 0,
        }

    input_tokens = _get_nested_int(usage, "input_tokens") or _get_nested_int(
        usage, "prompt_tokens"
    )
    output_tokens = _get_nested_int(usage, "output_tokens") or _get_nested_int(
        usage, "completion_tokens"
    )
    cached_input_tokens = _get_nested_int(
        usage, "input_tokens_details", "cached_tokens"
    ) or _get_nested_int(usage, "prompt_tokens_details", "cached_tokens")
    cache_write_tokens = _get_nested_int(
        usage, "input_tokens_details", "cache_write_tokens"
    ) or _get_nested_int(usage, "prompt_tokens_details", "cache_write_tokens")
    tool_usage = getattr(response, "tool_usage", None)
    web_search_calls = _get_nested_int(
        tool_usage, "web_search", "num_requests"
    ) or _get_nested_int(tool_usage, "web_search", "requests")
    token_counts = {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_write_tokens": cache_write_tokens,
        "output_tokens": output_tokens,
        "web_search_calls": web_search_calls,
    }

    token_cost = estimate_token_cost(
        model,
        TokenCounts(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_tokens=cache_write_tokens,
        ),
    )
    if token_cost is None:
        return None, token_counts

    tool_cost = web_search_calls * WEB_SEARCH_COST_PER_CALL
    return token_cost + tool_cost, token_counts


def _read_daily_total(cost_log_path: Path, date_text: str) -> float:
    if not cost_log_path.exists():
        return 0.0

    total = 0.0
    with cost_log_path.open(encoding="utf-8") as cost_log:
        for line in cost_log:
            if not line.startswith(f"{date_text}T") or " cost_usd=" not in line:
                continue
            cost_text = line.split(" cost_usd=", maxsplit=1)[1].split()[0]
            try:
                total += float(cost_text)
            except ValueError:
                continue
    return total


def _append_cost_log(
    *,
    model: str,
    response: Any,
    cost_log_path: Path = COST_LOG_PATH,
) -> float | None:
    timestamp = datetime.now().astimezone()
    date_text = timestamp.date().isoformat()
    cost, token_counts = _estimate_response_cost(response, model)
    daily_total = _read_daily_total(cost_log_path, date_text) + (cost or 0.0)
    cost_log_path.parent.mkdir(parents=True, exist_ok=True)

    if cost is None:
        cost_text = "unavailable"
    else:
        cost_text = f"{cost:.8f}"

    with cost_log_path.open("a", encoding="utf-8") as cost_log:
        cost_log.write(
            f"{timestamp.isoformat()} model={model} "
            f"input_tokens={token_counts['input_tokens']} "
            f"cached_input_tokens={token_counts['cached_input_tokens']} "
            f"cache_write_tokens={token_counts['cache_write_tokens']} "
            f"output_tokens={token_counts['output_tokens']} "
            f"web_search_calls={token_counts['web_search_calls']} "
            f"cost_usd={cost_text} daily_total_usd={daily_total:.8f}\n"
        )
    return cost


def _append_daily_cost_summary(
    cost_log_path: Path = COST_LOG_PATH,
    *,
    label: str = "run_end",
) -> None:
    timestamp = datetime.now().astimezone()
    date_text = timestamp.date().isoformat()
    daily_total = _read_daily_total(cost_log_path, date_text)
    cost_log_path.parent.mkdir(parents=True, exist_ok=True)
    with cost_log_path.open("a", encoding="utf-8") as cost_log:
        cost_log.write(
            f"{timestamp.isoformat()} {label} daily_total_usd={daily_total:.8f}\n"
        )


def _load_start_json(start_json: str | Path | None) -> PtxKernel | None:
    """Load and validate an optional starting implementation.

    Args:
        start_json: Inline candidate JSON, a JSON file path, or ``None``.

    Returns:
        The validated starting candidate, or ``None`` when one was not supplied.

    Raises:
        OSError: If a supplied JSON file cannot be read.
        ValueError: If the JSON does not contain a valid PTX candidate.
    """
    if start_json is None:
        return None

    value = str(start_json)
    serialized_candidate = (
        value
        if value.lstrip().startswith("{")
        else Path(value).read_text(encoding="utf-8")
    )
    loaded_data = json.loads(serialized_candidate)
    if not isinstance(loaded_data, dict):
        raise ValueError("Starting candidate JSON must contain an object.")

    candidate_data = loaded_data.get("payload", loaded_data)
    if not isinstance(candidate_data, dict):
        raise ValueError("Starting candidate payload must contain an object.")

    candidate_data = {
        key: value
        for key, value in candidate_data.items()
        if key in PtxKernel.model_fields
    }
    return PtxKernel.model_validate(candidate_data)


def _final_path(trace_path: Path) -> Path:
    """Return the final-answer path corresponding to a trace path.

    Args:
        trace_path: Full response trace destination.

    Returns:
        Sibling path with ``_final`` appended to the trace stem.
    """
    return trace_path.with_name(f"{trace_path.stem}_final.json")


def _create_response_with_retries(client: OpenAI, kwargs: dict[str, Any]) -> Any:
    """Create a response, retrying transient skill lookup misses."""
    for attempt in range(RESPONSE_RETRY_ATTEMPTS):
        try:
            return client.responses.create(**kwargs)
        except NotFoundError:
            if attempt == RESPONSE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)

    raise RuntimeError("Response retry loop ended unexpectedly.")


def _get_field(value: Any, field_name: str) -> Any:
    """Return a field from either an SDK object or a dictionary."""
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _format_tool_call(output_item: Any) -> str | None:
    """Return a concise tool-call description for a Responses output item."""
    item_type = _get_field(output_item, "type")
    if not isinstance(item_type, str) or not item_type.endswith("_call"):
        return None

    tool_name = (
        _get_field(output_item, "name")
        or _get_field(output_item, "tool_name")
        or _get_field(output_item, "server_label")
        or item_type.removesuffix("_call")
    )
    call_id = _get_field(output_item, "call_id") or _get_field(output_item, "id")
    status = _get_field(output_item, "status")

    details = [f"tool={tool_name}"]
    if call_id is not None:
        details.append(f"id={call_id}")
    if status is not None:
        details.append(f"status={status}")
    return ", ".join(details)


def _print_tool_calls(response: Any) -> None:
    """Print each tool call reported in a Responses API response."""
    output_items = _get_field(response, "output")
    if output_items is None:
        return

    for output_item in output_items:
        tool_call = _format_tool_call(output_item)
        if tool_call is not None:
            print(f"=== LLM called tool: {tool_call} ===", flush=True)


def _message_output_text(output_item: Any) -> str | None:
    """Return concatenated output text for one assistant message item."""
    if _get_field(output_item, "type") != "message":
        return None

    content_items = _get_field(output_item, "content")
    if content_items is None:
        return None

    text_parts = [
        text
        for content_item in content_items
        if _get_field(content_item, "type") == "output_text"
        for text in [_get_field(content_item, "text")]
        if isinstance(text, str)
    ]
    if not text_parts:
        return None
    return "".join(text_parts)


def _response_json_text(response: Any) -> str:
    """Return the final structured JSON text from a Responses API response.

    Raises:
        ValueError: If the response has no message output text.
    """
    output_items = _get_field(response, "output")
    if output_items is None:
        output_text = _get_field(response, "output_text")
        if isinstance(output_text, str) and output_text:
            return output_text
        raise ValueError("Response did not contain output text.")

    fallback_text = None
    for output_item in output_items:
        output_text = _message_output_text(output_item)
        if output_text is None:
            continue
        if _get_field(output_item, "phase") == "final_answer":
            return output_text
        fallback_text = output_text

    if fallback_text is None:
        raise ValueError("Response did not contain message output text.")
    return fallback_text


def _candidate_json(candidate: PtxKernel) -> str:
    """Return canonical JSON for a PTX candidate."""
    return candidate.model_dump_json(exclude_none=False, indent=2)


def _candidate_from_evaluation(evaluation: Any) -> PtxKernel:
    """Return a validated PTX candidate from an evaluator result."""
    payload = Payload.from_input(evaluation.payload).to_launch_dict()
    return PtxKernel.model_validate(payload)


def _evaluation_summary(evaluation: Any, *, include_ptx: bool) -> str:
    """Format a compact evaluator result summary for prompts and traces."""
    payload = _candidate_from_evaluation(evaluation).model_dump(exclude_none=False)
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
            default=_json_default,
        )[-4000:]
    if evaluation.compile_error:
        fields["compile_error"] = evaluation.compile_error[-2000:]
    if evaluation.timing_error:
        fields["timing_error"] = evaluation.timing_error[-2000:]

    return json.dumps(fields, indent=2, default=_json_default)


def _build_improvement_prompt(
    base_prompt: str,
    best_evaluation: Any,
    recent_evaluations: list[Any],
) -> str:
    """Build a compact prompt asking for three targeted improvement ideas."""
    recent_block = "\n\n".join(
        _evaluation_summary(evaluation, include_ptx=False)
        for evaluation in recent_evaluations[-6:]
    )
    if not recent_block:
        recent_block = "None yet."

    return f"""{base_prompt}

## Planning Override

For this response only, do not generate PTX and ignore the output contract above.
Return only the structured three-idea improvement plan requested below.

## Current Best Verified Candidate

{_evaluation_summary(best_evaluation, include_ptx=True)}

## Recent Candidate Outcomes

{recent_block}

## Planning Task

First decide how to improve the current best kernel. Return exactly three
specific, independent improvement ideas. Each idea must change one meaningful
performance factor only, explain why it could help, and be concrete enough to
generate one PTX candidate from it. Do not include PTX in this planning answer.

Each idea must be a localized micro-change to the current best PTX, not a
rewrite. The candidate must preserve every part of the kernel that is not
directly required by the proposed improvement: tiling, micro-tile shape,
unrolling structure, register accumulators, shared-memory staging, store
pattern, predicates, and algorithm. Do not replace the kernel with generic
loops, local-memory accumulator arrays, or a different implementation strategy.
""".strip()


def _build_candidate_prompt(
    base_prompt: str,
    best_evaluation: Any,
    idea: dict[str, str],
) -> str:
    """Build a prompt for one focused candidate derived from one idea."""
    return f"""{base_prompt}

## Current Best Verified Candidate

{_evaluation_summary(best_evaluation, include_ptx=True)}

## Single Improvement To Try

Name: {idea["name"]}
Rationale: {idea["rationale"]}
Instruction: {idea["instruction"]}

Generate exactly one PTX candidate by applying only this improvement to the
current best. Preserve correctness and the required output schema.

This must be a micro-edit of the current best PTX. Preserve its tiling strategy,
micro-tile shape, manual unroll structure, register accumulators,
shared-memory staging, synchronization strategy, predicate/store pattern,
launch shape, and PTX signature unless the single improvement explicitly
requires touching one of those items. Do not generate a simpler replacement:
no generic scalar i/j/k loops, no local-memory accumulator arrays, no shorter
basic implementation, and no clean-room rewrite. The output should be
recognizably the same optimized PTX plus the requested improvement.

If the idea is an addressing/layout tweak, such as shared-memory stride,
padding, or skew, change only the relevant shared-memory allocation and address
arithmetic. Leave the compute microkernel and stores intact. If the idea turns
out not to apply, make the smallest useful related micro-change instead.

Before returning the final JSON for this candidate, call the available
`triton_ptx` tool to compile, verify, and benchmark your attempted improvement.
If it fails compilation or correctness, repair the same attempted candidate
using the diagnostics and call `triton_ptx` again. Return only the fastest
verified version you actually tested. If no repair passes, return the closest
repaired candidate you tested so the outer loop can record diagnostics.
""".strip()


def _build_repair_prompt(
    base_prompt: str,
    best_evaluation: Any,
    failed_evaluation: Any,
    idea: dict[str, str],
    *,
    repair_index: int,
    max_repair_attempts: int,
) -> str:
    """Build a prompt that repairs one failed candidate in isolation."""
    return f"""{base_prompt}

## Current Best Verified Candidate

{_evaluation_summary(best_evaluation, include_ptx=True)}

## Original Improvement Being Tried

Name: {idea["name"]}
Rationale: {idea["rationale"]}
Instruction: {idea["instruction"]}

## Failed Candidate And Diagnostics

{_evaluation_summary(failed_evaluation, include_ptx=True)}

## Repair Task

This is repair attempt {repair_index} of {max_repair_attempts}. Repair the
failed candidate above, not the current best candidate from scratch.

Make the smallest concrete change needed to fix the compile, verification, or
runtime failure while preserving the original improvement idea. Keep the same
PTX signature, launch metadata keys, tiling strategy, micro-tile shape,
shared-memory staging, synchronization strategy, predicate/store pattern, and
manual unroll structure unless the diagnostic proves one of those exact parts
is the bug.

Call the available `triton_ptx` tool before returning. If the repaired candidate
still fails, use the new diagnostic to make one more minimal repair while
remaining within this attempt. Return only a JSON candidate that you actually
tested with the tool; prefer the fastest verified repair. If no repair passes,
return the closest tested repair so the outer loop can record its diagnostics.
""".strip()


def _build_initial_repair_prompt(
    base_prompt: str,
    failed_evaluation: Any,
    *,
    repair_index: int,
    max_repair_attempts: int,
) -> str:
    """Build a prompt that repairs the first generated candidate."""
    return f"""{base_prompt}

## Failed Initial Candidate And Diagnostics

{_evaluation_summary(failed_evaluation, include_ptx=True)}

## Initial Candidate Repair Task

This is initial candidate repair attempt {repair_index} of
{max_repair_attempts}. Repair the failed candidate above rather than generating
a fresh implementation from scratch.

Make the smallest concrete change needed to fix the compile, verification, or
runtime failure. Keep the same PTX signature, launch metadata keys, tiling
strategy, micro-tile shape, shared-memory staging, synchronization strategy,
predicate/store pattern, and manual unroll structure unless the diagnostic
proves one of those exact parts is the bug.

Call the available `triton_ptx` tool before returning. If the repaired candidate
still fails, use the new diagnostic to make one more minimal repair while
remaining within this attempt. Return only a JSON candidate that you actually
tested with the tool. If no repair passes, return the closest tested repair so
the outer loop can record its diagnostics.
""".strip()


def _request_json(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    response_format: dict[str, Any],
    reasoning_effort: str | None,
    tools: list[dict[str, Any]],
) -> tuple[Any, float | None]:
    """Send one compact Responses API request and log its cost."""
    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
        "input": [{"role": "user", "content": prompt}],
        "tools": tools,
        "text": {"format": response_format},
    }
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    response = _create_response_with_retries(client, kwargs)
    _print_tool_calls(response)
    return response, _append_cost_log(model=model, response=response)


def _record_prompt(
    responses: list[dict[str, Any]],
    trace_path: Path | None,
    *,
    prompt_name: str,
    prompt: str,
    round_index: int,
    candidate_index: int | None = None,
    idea: dict[str, str] | None = None,
) -> None:
    """Append an explicit prompt event to the trace."""
    responses.append(
        {
            "type": "prompt",
            "prompt_name": prompt_name,
            "round_index": round_index,
            "candidate_index": candidate_index,
            "idea": idea,
            "prompt": prompt,
        }
    )
    _write_trace(trace_path, responses)


def _evaluate_and_record(
    evaluator: TritonPTXCandidateEvaluator,
    candidate: PtxKernel,
    responses: list[dict[str, Any]],
    trace_path: Path | None,
    *,
    round_index: int,
    candidate_index: int,
    idea: dict[str, str] | None,
) -> Any:
    """Evaluate one candidate and append a trace record."""
    evaluated_candidate = evaluator.evaluate(
        Payload.from_input(candidate.model_dump(exclude_none=False))
    )
    print(
        f"Candidate result: compiles={evaluated_candidate.compiles}, "
        f"correct={evaluated_candidate.correct}, "
        f"p50={evaluated_candidate.p50}, "
        f"speedup={evaluated_candidate.speedup_vs_triton}, "
        f"message={evaluated_candidate.message}",
        flush=True,
    )
    responses.append(
        {
            "round_index": round_index,
            "candidate_index": candidate_index,
            "idea": idea,
            "evaluation": json.loads(evaluated_candidate.to_json(indent=2)),
        }
    )
    _write_trace(trace_path, responses)
    return evaluated_candidate


def _should_repair_candidate(evaluation: Any) -> bool:
    """Return whether a candidate failure should trigger an LLM repair attempt."""
    if evaluation.passed:
        return False
    if not evaluation.compiles:
        return True
    if not evaluation.correct:
        return True
    return bool(evaluation.timing_error)


def _generate_tested_candidate(
    client: OpenAI,
    evaluator: TritonPTXCandidateEvaluator,
    responses: list[dict[str, Any]],
    trace_path: Path | None,
    *,
    model: str,
    optimization_prompt: str,
    best_evaluation: Any,
    idea: dict[str, str],
    round_index: int,
    candidate_index: int,
    reasoning_effort: str | None,
    tools: list[dict[str, Any]],
    max_repair_attempts: int,
) -> Any:
    """Generate, evaluate, and minimally repair one idea before returning it."""
    candidate_prompt = _build_candidate_prompt(
        optimization_prompt,
        best_evaluation,
        idea,
    )
    _record_prompt(
        responses,
        trace_path,
        prompt_name="candidate",
        prompt=candidate_prompt,
        round_index=round_index,
        candidate_index=candidate_index,
        idea=idea,
    )
    response, _ = _request_json(
        client,
        model=model,
        prompt=candidate_prompt,
        response_format=RESPONSE_FORMAT,
        reasoning_effort=reasoning_effort,
        tools=tools,
    )
    responses.append(response.model_dump(mode="json"))
    _write_trace(trace_path, responses)
    candidate = PtxKernel.model_validate_json(_response_json_text(response))
    print(
        "--- Generated candidate ---\n"
        f"{_candidate_json(candidate)}\n"
        "--- End generated candidate ---",
        flush=True,
    )
    best_attempt = _evaluate_and_record(
        evaluator,
        candidate,
        responses,
        trace_path,
        round_index=round_index,
        candidate_index=candidate_index,
        idea=idea,
    )

    for repair_index in range(1, max_repair_attempts + 1):
        if not _should_repair_candidate(best_attempt):
            break

        print(
            f"=== TTS round {round_index}: repairing candidate "
            f"{candidate_index}/3 attempt {repair_index}/"
            f"{max_repair_attempts} ===",
            flush=True,
        )
        repair_prompt = _build_repair_prompt(
            optimization_prompt,
            best_evaluation,
            best_attempt,
            idea,
            repair_index=repair_index,
            max_repair_attempts=max_repair_attempts,
        )
        _record_prompt(
            responses,
            trace_path,
            prompt_name="candidate_repair",
            prompt=repair_prompt,
            round_index=round_index,
            candidate_index=candidate_index,
            idea=idea,
        )
        response, _ = _request_json(
            client,
            model=model,
            prompt=repair_prompt,
            response_format=RESPONSE_FORMAT,
            reasoning_effort=reasoning_effort,
            tools=tools,
        )
        responses.append(response.model_dump(mode="json"))
        _write_trace(trace_path, responses)
        repaired_candidate = PtxKernel.model_validate_json(
            _response_json_text(response)
        )
        print(
            "--- Repaired candidate ---\n"
            f"{_candidate_json(repaired_candidate)}\n"
            "--- End repaired candidate ---",
            flush=True,
        )
        repaired_evaluation = _evaluate_and_record(
            evaluator,
            repaired_candidate,
            responses,
            trace_path,
            round_index=round_index,
            candidate_index=candidate_index,
            idea=idea,
        )
        if repaired_evaluation.passed and (
            not best_attempt.passed or repaired_evaluation.p50 < best_attempt.p50
        ):
            best_attempt = repaired_evaluation
        elif not best_attempt.passed:
            best_attempt = repaired_evaluation

    return best_attempt


def _repair_initial_candidate(
    client: OpenAI,
    evaluator: TritonPTXCandidateEvaluator,
    responses: list[dict[str, Any]],
    trace_path: Path | None,
    *,
    model: str,
    optimization_prompt: str,
    initial_evaluation: Any,
    reasoning_effort: str | None,
    tools: list[dict[str, Any]],
    max_repair_attempts: int,
) -> Any:
    """Repair the initial candidate before the optimization rounds begin."""
    best_attempt = initial_evaluation
    for repair_index in range(1, max_repair_attempts + 1):
        if not _should_repair_candidate(best_attempt):
            break

        print(
            "=== Repairing initial candidate attempt "
            f"{repair_index}/{max_repair_attempts} ===",
            flush=True,
        )
        repair_prompt = _build_initial_repair_prompt(
            optimization_prompt,
            best_attempt,
            repair_index=repair_index,
            max_repair_attempts=max_repair_attempts,
        )
        _record_prompt(
            responses,
            trace_path,
            prompt_name="initial_candidate_repair",
            prompt=repair_prompt,
            round_index=0,
            candidate_index=0,
        )
        response, _ = _request_json(
            client,
            model=model,
            prompt=repair_prompt,
            response_format=RESPONSE_FORMAT,
            reasoning_effort=reasoning_effort,
            tools=tools,
        )
        responses.append(response.model_dump(mode="json"))
        _write_trace(trace_path, responses)
        repaired_candidate = PtxKernel.model_validate_json(
            _response_json_text(response)
        )
        print(
            "--- Repaired initial candidate ---\n"
            f"{_candidate_json(repaired_candidate)}\n"
            "--- End repaired initial candidate ---",
            flush=True,
        )
        repaired_evaluation = _evaluate_and_record(
            evaluator,
            repaired_candidate,
            responses,
            trace_path,
            round_index=0,
            candidate_index=0,
            idea=None,
        )
        if repaired_evaluation.passed and (
            not best_attempt.passed or repaired_evaluation.p50 < best_attempt.p50
        ):
            best_attempt = repaired_evaluation
        elif not best_attempt.passed:
            best_attempt = repaired_evaluation

    return best_attempt


def run_agent_loop(
    kernel_name: str,
    *,
    model: str = "gpt-5",
    max_tool_rounds: int = 3,
    max_repair_attempts: int = DEFAULT_REPAIR_ATTEMPTS,
    reasoning_effort: str | None = "medium",
    trace_path: Path | None = Path("trace.json"),
    start_json: str | Path | None = None,
    use_ptx_skill: bool = True,
    skill_dir: Path | None = None,
    skill_subdir: str = DEFAULT_PTX_SKILL_SUBDIR,
) -> str:
    """Optimize a kernel with compact explicit test-time scaling rounds.

    Args:
        kernel_name: Registered Triton PTX kernel class name.
        model: OpenAI model identifier.
        max_tool_rounds: Maximum improvement rounds. Each round evaluates three
            targeted candidates.
        max_repair_attempts: Maximum outer repair attempts per failed candidate.
        reasoning_effort: Optional reasoning effort.
        trace_path: Optional response trace destination.
        start_json: Optional inline JSON or JSON file containing the implementation
            from which optimization should continue.
        use_ptx_skill: Whether to upload and mount the PTX skill.
        skill_dir: Optional local skill directory. When omitted, the skill is read
            from the checked-out PTX skill submodule.
        skill_subdir: Submodule-relative skill directory used as the skill package.

    Returns:
        Final model response text.

    Raises:
        RuntimeError: If the model does not finish within the round limit.
    """
    if max_repair_attempts < 0:
        raise ValueError("max_repair_attempts must be non-negative.")

    evaluator = TritonPTXCandidateEvaluator(resolve_kernel(kernel_name))
    client = OpenAI()
    skill_id = None
    if use_ptx_skill:
        skill_id = _upload_skill(
            client,
            skill_dir=skill_dir,
            skill_subdir=skill_subdir,
        )
        print(f"=== Uploaded PTX skill {skill_id} ===", flush=True)
    tools = _build_tools(skill_id)
    starting_candidate = _load_start_json(start_json)
    optimization_prompt = (
        build_initial_prompt(kernel_name)
        if starting_candidate is None
        else build_continuation_prompt(kernel_name)
    )
    responses: list[dict[str, Any]] = []
    recent_evaluations: list[Any] = []
    wrote_daily_summary = False

    def write_daily_summary() -> None:
        nonlocal wrote_daily_summary
        if not wrote_daily_summary:
            _append_daily_cost_summary()
            wrote_daily_summary = True

    atexit.register(write_daily_summary)

    try:
        if starting_candidate is None:
            print(
                "=== Generating initial candidate ===",
                flush=True,
            )
            _record_prompt(
                responses,
                trace_path,
                prompt_name="initial_candidate",
                prompt=optimization_prompt,
                round_index=0,
            )
            response, _ = _request_json(
                client,
                model=model,
                prompt=optimization_prompt,
                response_format=RESPONSE_FORMAT,
                reasoning_effort=reasoning_effort,
                tools=tools,
            )
            responses.append(response.model_dump(mode="json"))
            _write_trace(trace_path, responses)
            starting_candidate = PtxKernel.model_validate_json(
                _response_json_text(response)
            )

        best_evaluation = _evaluate_and_record(
            evaluator,
            starting_candidate,
            responses,
            trace_path,
            round_index=0,
            candidate_index=0,
            idea=None,
        )
        best_evaluation = _repair_initial_candidate(
            client,
            evaluator,
            responses,
            trace_path,
            model=model,
            optimization_prompt=optimization_prompt,
            initial_evaluation=best_evaluation,
            reasoning_effort=reasoning_effort,
            tools=tools,
            max_repair_attempts=max_repair_attempts,
        )
        if not best_evaluation.passed:
            raise RuntimeError("Initial candidate must compile and pass verification.")

        for round_index in range(1, max_tool_rounds + 1):
            print(
                f"=== TTS round {round_index}/{max_tool_rounds}: planning three "
                "focused improvements ===",
                flush=True,
            )
            plan_prompt = _build_improvement_prompt(
                optimization_prompt,
                best_evaluation,
                recent_evaluations,
            )
            _record_prompt(
                responses,
                trace_path,
                prompt_name="improvement_plan",
                prompt=plan_prompt,
                round_index=round_index,
            )
            request_start = perf_counter()
            response, cost = _request_json(
                client,
                model=model,
                prompt=plan_prompt,
                response_format=IMPROVEMENT_PLAN_FORMAT,
                reasoning_effort=reasoning_effort,
                tools=tools,
            )
            request_duration = perf_counter() - request_start
            cost_message = "cost unavailable" if cost is None else f"cost ${cost:.6f}"
            print(
                f"=== TTS round {round_index}: received improvement plan in "
                f"{request_duration:.3f}s; {cost_message} ===",
                flush=True,
            )
            responses.append(response.model_dump(mode="json"))
            _write_trace(trace_path, responses)
            ideas = json.loads(_response_json_text(response))["improvements"]
            round_evaluations = []

            for candidate_index, idea in enumerate(ideas, start=1):
                print(
                    f"=== TTS round {round_index}: generating candidate "
                    f"{candidate_index}/3 for {idea['name']!r} ===",
                    flush=True,
                )
                evaluated_candidate = _generate_tested_candidate(
                    client,
                    evaluator,
                    responses,
                    trace_path,
                    model=model,
                    optimization_prompt=optimization_prompt,
                    best_evaluation=best_evaluation,
                    idea=idea,
                    round_index=round_index,
                    candidate_index=candidate_index,
                    reasoning_effort=reasoning_effort,
                    tools=tools,
                    max_repair_attempts=max_repair_attempts,
                )
                recent_evaluations.append(evaluated_candidate)
                round_evaluations.append(evaluated_candidate)

            passed_evaluations = [
                evaluation for evaluation in round_evaluations if evaluation.passed
            ]
            if passed_evaluations:
                round_best = min(passed_evaluations, key=lambda result: result.p50)
                if round_best.p50 < best_evaluation.p50:
                    best_evaluation = round_best
                    print(
                        f"=== TTS round {round_index}: new best p50="
                        f"{best_evaluation.p50}, speedup="
                        f"{best_evaluation.speedup_vs_triton} ===",
                        flush=True,
                    )
                else:
                    print(
                        f"=== TTS round {round_index}: no faster verified "
                        "candidate found ===",
                        flush=True,
                    )
            else:
                print(
                    f"=== TTS round {round_index}: no candidate passed "
                    "verification ===",
                    flush=True,
                )

        final_candidate = _candidate_from_evaluation(best_evaluation)
        final_payload = final_candidate.model_dump(exclude_none=True)
        final_payload["speedup"] = best_evaluation.speedup_vs_triton
        final_payload["p50"] = best_evaluation.p50
        final_json = json.dumps(final_payload, indent=2, default=_json_default)
        if trace_path is not None:
            _final_path(trace_path).write_text(f"{final_json}\n", encoding="utf-8")
        write_daily_summary()
        return final_json
    finally:
        write_daily_summary()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Optimize PTX with an OpenAI agent.")
    parser.add_argument("kernel", help="Kernel class name, for example AddKernel.")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-tool-rounds", type=int, default=5)
    parser.add_argument(
        "--max-repair-attempts",
        type=int,
        default=DEFAULT_REPAIR_ATTEMPTS,
        help="Maximum outer LLM repair attempts per failed candidate.",
    )
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--trace-path", type=Path, default=Path("trace.json"))
    parser.add_argument(
        "--start-json",
        help="Inline candidate JSON or path to a candidate JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the PTX optimization command."""
    args = parse_args()
    print(
        run_agent_loop(
            args.kernel,
            model=args.model,
            max_tool_rounds=args.max_tool_rounds,
            max_repair_attempts=args.max_repair_attempts,
            reasoning_effort=(
                None if args.reasoning_effort == "none" else args.reasoning_effort
            ),
            trace_path=args.trace_path,
            start_json=args.start_json,
        )
    )


if __name__ == "__main__":
    main()
