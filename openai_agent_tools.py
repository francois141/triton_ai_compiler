from __future__ import annotations

import argparse
import atexit
import json
import sys
from pathlib import Path
from time import perf_counter

from openai import OpenAI

LOCAL_TRITON_PTX_ROOT = Path(__file__).resolve().parent / "triton_ptx"
if LOCAL_TRITON_PTX_ROOT.is_dir():
    sys.path.insert(0, str(LOCAL_TRITON_PTX_ROOT))

from prompts import build_continuation_prompt  # noqa: E402
from prompts.improvement import (  # noqa: E402
    build_candidate_prompt,
    build_improvement_prompt,
    build_initial_repair_prompt,
    build_repair_prompt,
)
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
    resolve_kernel,
)
from helpers.cost import append_daily_cost_summary  # noqa: E402
from skills import load_ptx  # noqa: E402
from helpers.response import request_json, response_json_text  # noqa: E402
from helpers.setup import (  # noqa: E402
    build_initial_prompt,
    build_tools,
    load_start_json,
)
from helpers.traces import (  # noqa: E402
    final_path,
    record_prompt,
    write_trace,
)


DEFAULT_REPAIR_ATTEMPTS = 4


def _candidate_json(candidate):
    return candidate.model_dump_json(exclude_none=False, indent=2)


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
    write_trace(trace_path, responses)
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
    record_prompt(
        responses,
        trace_path,
        prompt_name="candidate",
        prompt=candidate_prompt,
        round_index=round_index,
        candidate_index=candidate_index,
        idea=idea,
    )
    response, _ = request_json(
        client,
        model=model,
        prompt=candidate_prompt,
        response_format=PTX_KERNEL_RESPONSE_FORMAT,
        reasoning_effort=reasoning_effort,
        tools=tools,
    )
    responses.append(response.model_dump(mode="json"))
    write_trace(trace_path, responses)
    candidate = PtxKernel.model_validate_json(response_json_text(response))
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
        record_prompt(
            responses,
            trace_path,
            prompt_name="candidate_repair",
            prompt=repair_prompt,
            round_index=round_index,
            candidate_index=candidate_index,
            idea=idea,
        )
        response, _ = request_json(
            client,
            model=model,
            prompt=repair_prompt,
            response_format=PTX_KERNEL_RESPONSE_FORMAT,
            reasoning_effort=reasoning_effort,
            tools=tools,
        )
        responses.append(response.model_dump(mode="json"))
        write_trace(trace_path, responses)
        repaired_candidate = PtxKernel.model_validate_json(
            response_json_text(response)
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
        record_prompt(
            responses,
            trace_path,
            prompt_name="initial_candidate_repair",
            prompt=repair_prompt,
            round_index=0,
            candidate_index=0,
        )
        response, _ = request_json(
            client,
            model=model,
            prompt=repair_prompt,
            response_format=PTX_KERNEL_RESPONSE_FORMAT,
            reasoning_effort=reasoning_effort,
            tools=tools,
        )
        responses.append(response.model_dump(mode="json"))
        write_trace(trace_path, responses)
        repaired_candidate = PtxKernel.model_validate_json(
            response_json_text(response)
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
    tools = build_tools(skill_id)
    starting_candidate = load_start_json(start_json)
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
            record_prompt(
                responses,
                trace_path,
                prompt_name="initial_candidate",
                prompt=optimization_prompt,
                round_index=0,
            )
            response, _ = request_json(
                client,
                model=model,
                prompt=optimization_prompt,
                response_format=PTX_KERNEL_RESPONSE_FORMAT,
                reasoning_effort=reasoning_effort,
                tools=tools,
            )
            responses.append(response.model_dump(mode="json"))
            write_trace(trace_path, responses)
            starting_candidate = PtxKernel.model_validate_json(
                response_json_text(response)
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
            record_prompt(
                responses,
                trace_path,
                prompt_name="improvement_plan",
                prompt=plan_prompt,
                round_index=round_index,
            )
            request_start = perf_counter()
            response, cost = request_json(
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
            write_trace(trace_path, responses)
            ideas = json.loads(response_json_text(response))["improvements"]
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
            final_path(trace_path).write_text(f"{final_json}\n", encoding="utf-8")
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
