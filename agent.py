from __future__ import annotations

import argparse
import atexit
import json
from pathlib import Path
from time import perf_counter

from prompts.improvement import (
    build_candidate_prompt,
    build_failure_analysis_prompt,
    build_improvement_prompt,
    build_initial_repair_prompt,
    build_ncu_improvement_prompt,
    build_repair_prompt,
)
from utils.response_format import (
    FAILURE_ANALYSIS_RESPONSE_FORMAT,
    IMPROVEMENT_PLAN_RESPONSE_FORMAT,
    PTX_KERNEL_METADATA_RESPONSE_FORMAT,
    PTX_KERNEL_RESPONSE_FORMAT,
    FailureAnalysis,
    PtxKernel,
    PtxKernelMetadata,
)
from utils.evaluation import (
    candidate_from_evaluation,
    json_default,
)
from triton_ptx import Payload
from utils.cost import append_daily_cost_summary
from utils.providers import create_provider_session
from utils.response import response_json_text, verifier_for_kernel
from utils.setup import (
    build_initial_prompt,
    load_start_json,
    load_start_ptx,
    load_triton_generated_ptx,
)
from utils.traces import (
    create_trace_directory,
    record_generated_candidate,
    record_generated_json,
    record_prompt,
    write_trace,
)


def _candidate_json(candidate):
    return candidate.model_dump_json(exclude_none=False, indent=2)


def _candidate_from_patch_response(response, patched_candidate):
    if patched_candidate is None:
        raise ValueError("The model did not apply a PTX patch.")
    metadata = PtxKernelMetadata.model_validate_json(response_json_text(response))
    launch_dimensions = (
        metadata.num_threads_x,
        metadata.num_threads_y,
        metadata.num_threads_z,
    )
    patched_dimensions = (
        patched_candidate.num_threads_x,
        patched_candidate.num_threads_y,
        patched_candidate.num_threads_z,
    )
    if launch_dimensions != patched_dimensions:
        raise ValueError(
            "The final launch metadata must match the most recently patched "
            "and verified PTX candidate."
        )
    return PtxKernel(
        ptx=patched_candidate.ptx,
        num_threads_x=metadata.num_threads_x,
        num_threads_y=metadata.num_threads_y,
        num_threads_z=metadata.num_threads_z,
        difficulties=metadata.difficulties,
    )


def _evaluate_and_record(
    evaluator,
    candidate,
    responses,
    trace_path,
    *,
    round_index,
    candidate_index,
    idea,
    prompt_name,
    attempt_index,
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
    record_generated_candidate(
        trace_path,
        candidate,
        evaluated_candidate,
        prompt_name=prompt_name,
        round_index=round_index,
        attempt_index=attempt_index,
        candidate_index=candidate_index,
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


def _require_ncu_report(evaluation):
    ncu_report = evaluation.ncu_report
    if ncu_report.get("available") and ncu_report.get("summary"):
        return
    raise RuntimeError(
        "--ncu-decision requires a successful Nsight Compute report. "
        f"NCU error: {ncu_report.get('error', '')}"
    )


def _generate_tested_candidate(
    provider_session,
    evaluator,
    responses,
    trace_path,
    *,
    model,
    kernel_name,
    base_prompt,
    best_evaluation,
    idea,
    round_index,
    candidate_index,
    reasoning_effort,
    max_repair_attempts,
):
    candidate_prompt = build_candidate_prompt(
        base_prompt,
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
        speedup_vs_triton=best_evaluation.speedup_vs_triton,
    )
    response, _, patched_candidate = provider_session.request_json(
        model=model,
        prompt=candidate_prompt,
        response_format=PTX_KERNEL_METADATA_RESPONSE_FORMAT,
        reasoning_effort=reasoning_effort,
        kernel_name=kernel_name,
        current_candidate=candidate_from_evaluation(best_evaluation),
    )
    responses.append(response.model_dump(mode="json"))
    write_trace(trace_path, responses)
    candidate = _candidate_from_patch_response(response, patched_candidate)
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
        prompt_name="candidate",
        attempt_index=0,
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
        analysis_prompt = build_failure_analysis_prompt(
            base_prompt,
            best_attempt,
            repair_index=repair_index,
            max_repair_attempts=max_repair_attempts,
            idea=idea,
        )
        record_prompt(
            responses,
            trace_path,
            prompt_name="candidate_failure_analysis",
            prompt=analysis_prompt,
            round_index=round_index,
            attempt_index=repair_index,
            candidate_index=candidate_index,
            idea=idea,
            speedup_vs_triton=best_attempt.speedup_vs_triton,
        )
        analysis_response, _, _ = provider_session.request_json(
            model=model,
            prompt=analysis_prompt,
            response_format=FAILURE_ANALYSIS_RESPONSE_FORMAT,
            reasoning_effort=reasoning_effort,
            kernel_name=kernel_name,
        )
        responses.append(analysis_response.model_dump(mode="json"))
        write_trace(trace_path, responses)
        failure_analysis = FailureAnalysis.model_validate_json(
            response_json_text(analysis_response)
        ).model_dump()
        repair_prompt = build_repair_prompt(
            base_prompt,
            best_evaluation,
            best_attempt,
            idea,
            failure_analysis,
            repair_index=repair_index,
            max_repair_attempts=max_repair_attempts,
        )
        record_prompt(
            responses,
            trace_path,
            prompt_name="candidate_repair",
            prompt=repair_prompt,
            round_index=round_index,
            attempt_index=repair_index,
            candidate_index=candidate_index,
            idea=idea,
            speedup_vs_triton=best_attempt.speedup_vs_triton,
        )
        response, _, patched_candidate = provider_session.request_json(
            model=model,
            prompt=repair_prompt,
            response_format=PTX_KERNEL_METADATA_RESPONSE_FORMAT,
            reasoning_effort=reasoning_effort,
            kernel_name=kernel_name,
            current_candidate=candidate_from_evaluation(best_attempt),
        )
        responses.append(response.model_dump(mode="json"))
        write_trace(trace_path, responses)
        repaired_candidate = _candidate_from_patch_response(response, patched_candidate)
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
            prompt_name="candidate_repair",
            attempt_index=repair_index,
        )
        if repaired_evaluation.passed and (
            not best_attempt.passed or repaired_evaluation.p50 < best_attempt.p50
        ):
            best_attempt = repaired_evaluation
        elif not best_attempt.passed:
            best_attempt = repaired_evaluation

    return best_attempt


def _repair_initial_candidate(
    provider_session,
    evaluator,
    responses,
    trace_path,
    *,
    model,
    kernel_name,
    base_prompt,
    initial_evaluation,
    reasoning_effort,
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
        analysis_prompt = build_failure_analysis_prompt(
            base_prompt,
            best_attempt,
            repair_index=repair_index,
            max_repair_attempts=max_repair_attempts,
        )
        record_prompt(
            responses,
            trace_path,
            prompt_name="initial_candidate_failure_analysis",
            prompt=analysis_prompt,
            round_index=0,
            attempt_index=repair_index,
            candidate_index=0,
            speedup_vs_triton=best_attempt.speedup_vs_triton,
        )
        analysis_response, _, _ = provider_session.request_json(
            model=model,
            prompt=analysis_prompt,
            response_format=FAILURE_ANALYSIS_RESPONSE_FORMAT,
            reasoning_effort=reasoning_effort,
            kernel_name=kernel_name,
        )
        responses.append(analysis_response.model_dump(mode="json"))
        write_trace(trace_path, responses)
        failure_analysis = FailureAnalysis.model_validate_json(
            response_json_text(analysis_response)
        ).model_dump()
        repair_prompt = build_initial_repair_prompt(
            base_prompt,
            best_attempt,
            failure_analysis,
            repair_index=repair_index,
            max_repair_attempts=max_repair_attempts,
        )
        record_prompt(
            responses,
            trace_path,
            prompt_name="initial_candidate_repair",
            prompt=repair_prompt,
            round_index=0,
            attempt_index=repair_index,
            candidate_index=0,
            speedup_vs_triton=best_attempt.speedup_vs_triton,
        )
        response, _, patched_candidate = provider_session.request_json(
            model=model,
            prompt=repair_prompt,
            response_format=PTX_KERNEL_METADATA_RESPONSE_FORMAT,
            reasoning_effort=reasoning_effort,
            kernel_name=kernel_name,
            current_candidate=candidate_from_evaluation(best_attempt),
        )
        responses.append(response.model_dump(mode="json"))
        write_trace(trace_path, responses)
        repaired_candidate = _candidate_from_patch_response(response, patched_candidate)
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
            prompt_name="initial_candidate_repair",
            attempt_index=repair_index,
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
    model,
    provider="openai",
    max_tool_rounds,
    max_repair_attempts,
    reasoning_effort,
    trace_path,
    start_json=None,
    start_ptx=None,
    start_num_threads_x=128,
    start_num_threads_y=1,
    start_num_threads_z=1,
    start_triton_generated_ptx=False,
    ncu_decision=False,
):
    if max_repair_attempts < 0:
        raise ValueError("max_repair_attempts must be non-negative.")
    if (
        sum(
            value is not None and value is not False
            for value in (start_json, start_ptx, start_triton_generated_ptx)
        )
        > 1
    ):
        raise ValueError(
            "Use only one of --start-json, --start-ptx, or "
            "--start-triton-generated-ptx."
        )

    trace_path = create_trace_directory(
        trace_path,
        kernel_name,
        model,
        reasoning_effort,
    )
    print(f"=== Writing trace artifacts to {trace_path} ===", flush=True)

    evaluator = verifier_for_kernel(kernel_name)
    provider_session = create_provider_session(provider)

    starting_candidate = load_start_ptx(
        start_ptx,
        num_threads_x=start_num_threads_x,
        num_threads_y=start_num_threads_y,
        num_threads_z=start_num_threads_z,
    )
    if starting_candidate is None:
        starting_candidate = (
            load_start_json(start_json)
            if start_json is not None
            else (
                load_triton_generated_ptx(kernel_name)
                if start_triton_generated_ptx
                else None
            )
        )
    base_prompt = build_initial_prompt(kernel_name, evaluator.operator)
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
                prompt=base_prompt,
                round_index=0,
                speedup_vs_triton=None,
            )
            response, _, _ = provider_session.request_json(
                model=model,
                prompt=base_prompt,
                response_format=PTX_KERNEL_RESPONSE_FORMAT,
                reasoning_effort=reasoning_effort,
                kernel_name=kernel_name,
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
            prompt_name="initial_candidate",
            attempt_index=0,
        )
        best_evaluation = _repair_initial_candidate(
            provider_session,
            evaluator,
            responses,
            trace_path,
            model=model,
            kernel_name=kernel_name,
            base_prompt=base_prompt,
            initial_evaluation=best_evaluation,
            reasoning_effort=reasoning_effort,
            max_repair_attempts=max_repair_attempts,
        )
        if not best_evaluation.passed:
            raise RuntimeError("Initial candidate must compile and pass verification.")

        if ncu_decision:
            _require_ncu_report(best_evaluation)

        for round_index in range(1, max_tool_rounds + 1):
            if ncu_decision:
                _require_ncu_report(best_evaluation)
            print(
                f"=== TTS round {round_index}/{max_tool_rounds}: planning three "
                "ordered improvements ===",
                flush=True,
            )
            plan_prompt = (
                build_ncu_improvement_prompt(
                    base_prompt,
                    best_evaluation.ncu_report,
                    kernel_name,
                )
                if ncu_decision
                else build_improvement_prompt(
                    base_prompt,
                    best_evaluation,
                    recent_evaluations,
                    kernel_name,
                )
            )
            plan_prompt_name = (
                "ncu_improvement_plan" if ncu_decision else "improvement_plan"
            )
            record_prompt(
                responses,
                trace_path,
                prompt_name=plan_prompt_name,
                prompt=plan_prompt,
                round_index=round_index,
                speedup_vs_triton=best_evaluation.speedup_vs_triton,
            )
            request_start = perf_counter()
            response, cost, _ = provider_session.request_json(
                model=model,
                prompt=plan_prompt,
                response_format=IMPROVEMENT_PLAN_RESPONSE_FORMAT,
                reasoning_effort=reasoning_effort,
                kernel_name=kernel_name,
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
            record_generated_json(
                trace_path,
                {"improvements": ideas},
                prompt_name=plan_prompt_name,
                round_index=round_index,
                attempt_index=0,
                speedup_vs_triton=best_evaluation.speedup_vs_triton,
                include_speedup=False,
            )
            round_base = best_evaluation

            for candidate_index, idea in enumerate(ideas, start=1):
                print(
                    f"=== TTS round {round_index}: generating candidate "
                    f"{candidate_index}/3 for {idea['name']!r} ===",
                    flush=True,
                )
                evaluated_candidate = _generate_tested_candidate(
                    provider_session,
                    evaluator,
                    responses,
                    trace_path,
                    model=model,
                    kernel_name=kernel_name,
                    base_prompt=base_prompt,
                    best_evaluation=round_base,
                    idea=idea,
                    round_index=round_index,
                    candidate_index=candidate_index,
                    reasoning_effort=reasoning_effort,
                    max_repair_attempts=max_repair_attempts,
                )
                recent_evaluations.append(evaluated_candidate)
                if (
                    evaluated_candidate.passed
                    and evaluated_candidate.p50 < round_base.p50
                ):
                    round_base = evaluated_candidate
                    print(
                        f"=== TTS round {round_index}: accepted candidate "
                        f"{candidate_index}/3 as the base for remaining ideas; "
                        f"p50={round_base.p50}, speedup="
                        f"{round_base.speedup_vs_triton} ===",
                        flush=True,
                    )

            if round_base is best_evaluation:
                print(
                    f"=== TTS round {round_index}: no faster verified "
                    "candidate found ===",
                    flush=True,
                )
            else:
                best_evaluation = round_base
                print(
                    f"=== TTS round {round_index}: new best p50="
                    f"{best_evaluation.p50}, speedup="
                    f"{best_evaluation.speedup_vs_triton} ===",
                    flush=True,
                )

        final_candidate = candidate_from_evaluation(best_evaluation)
        final_payload = final_candidate.model_dump(exclude_none=True)
        final_payload["speedup"] = best_evaluation.speedup_vs_triton
        final_payload["p50"] = best_evaluation.p50
        final_json = json.dumps(final_payload, indent=2, default=json_default)
        final_speedup = best_evaluation.speedup_vs_triton
        (trace_path / f"final_speedup_vs_triton_{final_speedup:.4f}x.json").write_text(
            f"{final_json}\n",
            encoding="utf-8",
        )
        (trace_path / "final_candidate.ptx").write_text(
            final_candidate.ptx.rstrip() + "\n",
            encoding="utf-8",
        )
        write_daily_summary()
        return final_json
    finally:
        write_daily_summary()


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize PTX with an LLM agent.")
    parser.add_argument("kernel", help="Kernel class name, for example AddKernel.")
    parser.add_argument("--model", help="Model to use; defaults depend on --provider.")
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic"),
        default="openai",
        help="LLM API provider to use.",
    )
    parser.add_argument("--max-tool-rounds", type=int, default=3)
    parser.add_argument(
        "--max-repair-attempts",
        type=int,
        default=10,
        help="Maximum outer LLM repair attempts per failed candidate.",
    )
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument(
        "--ncu-decision",
        action="store_true",
        help="Plan improvements exclusively from the Nsight Compute report.",
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=Path("output_traces"),
        help="Directory where a dated per-kernel trace folder is created.",
    )
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--start-json",
        help="Inline candidate JSON or path to a candidate JSON file.",
    )
    start_group.add_argument(
        "--start-ptx",
        type=Path,
        help="Path to a PTX file to edit and optimize.",
    )
    start_group.add_argument(
        "--start-triton-generated-ptx",
        action="store_true",
        help="Start from the saved Triton-generated PTX for this kernel.",
    )
    parser.add_argument(
        "--start-num-threads-x",
        type=int,
        default=128,
        help="X launch dimension for --start-ptx (default: 128).",
    )
    parser.add_argument(
        "--start-num-threads-y",
        type=int,
        default=1,
        help="Y launch dimension for --start-ptx (default: 1).",
    )
    parser.add_argument(
        "--start-num-threads-z",
        type=int,
        default=1,
        help="Z launch dimension for --start-ptx (default: 1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = (
        args.model
        or {
            "openai": "gpt-5.6-sol",
            "anthropic": "claude-opus-4-8",
        }[args.provider]
    )
    print(
        run_agent_loop(
            args.kernel,
            model=model,
            provider=args.provider,
            max_tool_rounds=args.max_tool_rounds,
            max_repair_attempts=args.max_repair_attempts,
            reasoning_effort=(
                None if args.reasoning_effort == "none" else args.reasoning_effort
            ),
            trace_path=args.trace_path,
            start_json=args.start_json,
            start_ptx=args.start_ptx,
            start_num_threads_x=args.start_num_threads_x,
            start_num_threads_y=args.start_num_threads_y,
            start_num_threads_z=args.start_num_threads_z,
            start_triton_generated_ptx=args.start_triton_generated_ptx,
            ncu_decision=args.ncu_decision,
        )
    )


if __name__ == "__main__":
    main()
