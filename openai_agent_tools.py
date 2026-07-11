from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import NotFoundError, OpenAI

LOCAL_TRITON_PTX_ROOT = Path(__file__).resolve().parent / "triton_ptx"
if LOCAL_TRITON_PTX_ROOT.is_dir():
    sys.path.insert(0, str(LOCAL_TRITON_PTX_ROOT))

from prompts import (  # noqa: E402
    build_continuation_prompt,
    build_prompt_for_operator,
)
from prompts.improvement import (  # noqa: E402
    build_candidate_prompt,
    build_improvement_prompt,
    build_initial_repair_prompt,
    build_repair_prompt,
)
from prompts.system import system_prompt  # noqa: E402
from utils.response_format import (  # noqa: E402
    IMPROVEMENT_PLAN_RESPONSE_FORMAT,
    PTX_KERNEL_RESPONSE_FORMAT,
    PtxKernel,
)
from utils.evaluation import (  # noqa: E402
    candidate_from_evaluation,
    json_default,
)
from triton_ptx import (  # noqa: E402
    Payload,
    TritonPTXCandidateEvaluator,
    dump_kernel_ptx,
    get_ptx_system_config,
    parse_ptx_signature,
    resolve_kernel,
)
from helpers.cost import append_cost_log, append_daily_cost_summary  # noqa: E402
from skills import load_ptx  # noqa: E402


RESPONSE_RETRY_ATTEMPTS = 6
DEFAULT_REPAIR_ATTEMPTS = 4


def _build_tools(skill_id):
    tools = []#[{"type": "web_search"}]
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


def build_initial_prompt(kernel_name):
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
    return f"{prompt}"


def _write_trace(trace_path, events):
    if trace_path is not None:
        trace_path.write_text(
            json.dumps(events, indent=2, default=json_default),
            encoding="utf-8",
        )


def _load_start_json(start_json):
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


def _final_path(trace_path):
    return trace_path.with_name(f"{trace_path.stem}_final.json")


def _create_response_with_retries(client, kwargs):
    for attempt in range(RESPONSE_RETRY_ATTEMPTS):
        try:
            return client.responses.create(**kwargs)
        except NotFoundError:
            if attempt == RESPONSE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)

    raise RuntimeError("Response retry loop ended unexpectedly.")


def _get_field(value, field_name):
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _format_tool_call(output_item):
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


def _print_tool_calls(response):
    output_items = _get_field(response, "output")
    if output_items is None:
        return

    for output_item in output_items:
        tool_call = _format_tool_call(output_item)
        if tool_call is not None:
            print(f"=== LLM called tool: {tool_call} ===", flush=True)


def _message_output_text(output_item):
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


def _response_json_text(response):
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


def _candidate_json(candidate):
    return candidate.model_dump_json(exclude_none=False, indent=2)


def _request_json(
    client,
    *,
    model,
    prompt,
    response_format,
    reasoning_effort,
    tools,
):
    kwargs = {
        "model": model,
        "instructions": system_prompt(),
        "input": [{"role": "user", "content": prompt}],
        "tools": tools,
        "text": {"format": response_format},
    }
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    response = _create_response_with_retries(client, kwargs)
    _print_tool_calls(response)
    return response, append_cost_log(model=model, response=response)


def _record_prompt(
    responses,
    trace_path,
    *,
    prompt_name,
    prompt,
    round_index,
    candidate_index = None,
    idea = None,
):
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
    evaluator,
    candidate,
    responses,
    trace_path,
    *,
    round_index,
    candidate_index,
    idea,
):
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


def _should_repair_candidate(evaluation):
    if evaluation.passed:
        return False
    if not evaluation.compiles:
        return True
    if not evaluation.correct:
        return True
    return bool(evaluation.timing_error)


def _generate_tested_candidate(
    client,
    evaluator,
    responses,
    trace_path,
    *,
    model,
    optimization_prompt,
    best_evaluation,
    idea,
    round_index,
    candidate_index,
    reasoning_effort,
    tools,
    max_repair_attempts,
):
    candidate_prompt = build_candidate_prompt(
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
        response_format=PTX_KERNEL_RESPONSE_FORMAT,
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
        repair_prompt = build_repair_prompt(
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
            response_format=PTX_KERNEL_RESPONSE_FORMAT,
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
    client,
    evaluator,
    responses,
    trace_path,
    *,
    model,
    optimization_prompt,
    initial_evaluation,
    reasoning_effort,
    tools,
    max_repair_attempts,
):
    best_attempt = initial_evaluation
    for repair_index in range(1, max_repair_attempts + 1):
        if not _should_repair_candidate(best_attempt):
            break

        print(
            "=== Repairing initial candidate attempt "
            f"{repair_index}/{max_repair_attempts} ===",
            flush=True,
        )
        repair_prompt = build_initial_repair_prompt(
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
            response_format=PTX_KERNEL_RESPONSE_FORMAT,
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
    kernel_name,
    *,
    model = "gpt-5.6-sol",
    max_tool_rounds = 5,
    max_repair_attempts = DEFAULT_REPAIR_ATTEMPTS,
    reasoning_effort = "medium",
    trace_path = Path("trace.json"),
    start_json = None,
):
    if max_repair_attempts < 0:
        raise ValueError("max_repair_attempts must be non-negative.")

    evaluator = TritonPTXCandidateEvaluator(resolve_kernel(kernel_name))
    client = OpenAI()
    skill_id = load_ptx(client)
    print(f"=== Uploaded PTX skill {skill_id} ===", flush=True)
    tools = _build_tools(skill_id)
    starting_candidate = _load_start_json(start_json)
    optimization_prompt = (
        build_initial_prompt(kernel_name)
        if starting_candidate is None
        else build_continuation_prompt(kernel_name)
    )
    responses = []
    recent_evaluations = []
    wrote_daily_summary = False

    def write_daily_summary():
        nonlocal wrote_daily_summary
        if not wrote_daily_summary:
            append_daily_cost_summary()
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
                response_format=PTX_KERNEL_RESPONSE_FORMAT,
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
            plan_prompt = build_improvement_prompt(
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
                response_format=IMPROVEMENT_PLAN_RESPONSE_FORMAT,
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

        final_candidate = candidate_from_evaluation(best_evaluation)
        final_payload = final_candidate.model_dump(exclude_none=True)
        final_payload["speedup"] = best_evaluation.speedup_vs_triton
        final_payload["p50"] = best_evaluation.p50
        final_json = json.dumps(final_payload, indent=2, default=json_default)
        if trace_path is not None:
            _final_path(trace_path).write_text(f"{final_json}\n", encoding="utf-8")
        write_daily_summary()
        return final_json
    finally:
        write_daily_summary()


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize PTX with an OpenAI agent.")
    parser.add_argument("kernel", help="Kernel class name, for example AddKernel.")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--max-tool-rounds", type=int, default=1)
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


def main():
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
