from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter
from pathlib import Path
from typing import Any

from openai import OpenAI

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

TOOLS = [
    {"type": "web_search"},
    {
        "type": "function",
        "name": "triton_ptx",
        "description": "Compile, test, verify, and benchmark one PTX candidate.",
        "parameters": {
            "type": "object",
            "properties": {"candidate": PAYLOAD_SCHEMA},
            "required": ["candidate"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


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
    return build_prompt_for_operator(
        kernel_cls,
        num_answers=1,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=signature,
    )


def _write_trace(trace_path: Path | None, events: list[dict[str, Any]]) -> None:
    if trace_path is not None:
        trace_path.write_text(json.dumps(events, indent=2), encoding="utf-8")


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
    candidate_data = json.loads(serialized_candidate)
    if not isinstance(candidate_data, dict):
        raise ValueError("Starting candidate JSON must contain an object.")
    candidate_data.pop("speedup", None)
    return PtxKernel.model_validate(candidate_data)


def _final_path(trace_path: Path) -> Path:
    """Return the final-answer path corresponding to a trace path.

    Args:
        trace_path: Full response trace destination.

    Returns:
        Sibling path with ``_final`` appended to the trace stem.
    """
    return trace_path.with_name(f"{trace_path.stem}_final.json")


def _candidate_key(candidate: PtxKernel) -> tuple[str, int, int | None, int | None]:
    """Return a stable lookup key for a candidate implementation.

    Args:
        candidate: Validated PTX candidate.

    Returns:
        PTX and launch dimensions identifying the candidate.
    """
    return (
        candidate.ptx,
        candidate.num_threads_x,
        candidate.num_threads_y,
        candidate.num_threads_z,
    )


def run_agent_loop(
    kernel_name: str,
    *,
    model: str = "gpt-5",
    max_tool_rounds: int = 20,
    reasoning_effort: str | None = "medium",
    trace_path: Path | None = Path("trace.json"),
    start_json: str | Path | None = None,
) -> str:
    """Optimize a kernel with native web search and one local evaluation tool.

    Args:
        kernel_name: Registered Triton PTX kernel class name.
        model: OpenAI model identifier.
        max_tool_rounds: Maximum model responses before stopping.
        reasoning_effort: Optional reasoning effort.
        trace_path: Optional response trace destination.
        start_json: Optional inline JSON or JSON file containing the implementation
            from which optimization should continue.

    Returns:
        Final model response text.

    Raises:
        RuntimeError: If the model does not finish within the round limit.
    """
    evaluator = TritonPTXCandidateEvaluator(resolve_kernel(kernel_name))
    client = OpenAI()
    initial_prompt = build_initial_prompt(kernel_name)
    starting_candidate = _load_start_json(start_json)
    if starting_candidate is not None:
        initial_prompt = (
            f"{initial_prompt}\n\n"
            "Continue optimizing from this current implementation. Evaluate it or "
            "improve it using the existing optimization loop:\n"
            f"{starting_candidate.model_dump_json(exclude_none=False, indent=2)}"
        )
    input_items: list[Any] = [{"role": "user", "content": initial_prompt}]
    previous_response_id: str | None = None
    responses: list[dict[str, Any]] = []
    measured_speedups: dict[tuple[str, int, int | None, int | None], float] = {}

    for round_index in range(1, max_tool_rounds + 1):
        print(
            f"=== Agent iteration {round_index}/{max_tool_rounds}: "
            "requesting the next optimization step ===",
            flush=True,
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "instructions": SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"),
            "input": input_items,
            "tools": TOOLS,
            "text": {"format": RESPONSE_FORMAT},
        }
        if previous_response_id is not None:
            kwargs["previous_response_id"] = previous_response_id
        if reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        request_start = perf_counter()
        response = client.responses.create(**kwargs)
        request_duration = perf_counter() - request_start
        print(
            f"=== Agent iteration {round_index}: received model response "
            f"in {request_duration:.3f}s ===",
            flush=True,
        )
        previous_response_id = response.id
        responses.append(response.model_dump(mode="json"))
        _write_trace(trace_path, responses)
        calls = [item for item in response.output if item.type == "function_call"]

        if not calls:
            print(
                f"=== Agent iteration {round_index}: optimization complete; "
                "returning the best verified candidate ===",
                flush=True,
            )
            final_candidate = PtxKernel.model_validate_json(response.output_text)
            speedup = measured_speedups.get(_candidate_key(final_candidate))
            if speedup is None:
                evaluated_final = evaluator.evaluate(
                    Payload.from_input(final_candidate.model_dump())
                )
                speedup = evaluated_final.speedup_vs_triton
            final_payload = final_candidate.model_dump(exclude_none=True)
            final_payload["speedup"] = speedup
            final_json = json.dumps(final_payload, indent=2)
            if trace_path is not None:
                _final_path(trace_path).write_text(f"{final_json}\n", encoding="utf-8")
            return final_json

        input_items = []
        for call_index, call in enumerate(calls, start=1):
            print(
                f"=== Agent iteration {round_index}: evaluating candidate "
                f"{call_index}/{len(calls)} ===",
                flush=True,
            )
            candidate = json.loads(call.arguments)["candidate"]
            validated_candidate = PtxKernel.model_validate(candidate)
            print(
                "--- Generated candidate ---\n"
                f"{json.dumps(candidate, indent=2)}\n"
                "--- End generated candidate ---",
                flush=True,
            )
            evaluated_candidate = evaluator.evaluate(Payload.from_input(candidate))
            measured_speedups[_candidate_key(validated_candidate)] = (
                evaluated_candidate.speedup_vs_triton
            )
            result_json = evaluated_candidate.to_json(indent=2)
            result = json.loads(result_json)
            print(
                f"Candidate result: compiles={evaluated_candidate.compiles}, "
                f"correct={evaluated_candidate.correct}, "
                f"p50={evaluated_candidate.p50}, "
                f"speedup={evaluated_candidate.speedup_vs_triton}, "
                f"message={evaluated_candidate.message}",
                flush=True,
            )
            if evaluated_candidate.verifier_report:
                verifier_report = evaluated_candidate.verifier_report
                print(
                    "Verifier report: "
                    f"status={verifier_report.get('status')}, "
                    f"size={verifier_report.get('size')}, "
                    f"iteration={verifier_report.get('iteration')}, "
                    f"num_wrong={verifier_report.get('num_wrong')}",
                    flush=True,
                )
            print(
                "--- Evaluator output ---\n"
                f"{result_json}\n"
                "--- End evaluator output ---",
                flush=True,
            )
            responses.append(
                {
                    "agent_iteration": round_index,
                    "candidate_index": call_index,
                    "evaluation": result,
                }
            )
            _write_trace(trace_path, responses)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result_json,
                }
            )

    raise RuntimeError(f"No final answer after {max_tool_rounds} tool rounds.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Optimize PTX with an OpenAI agent.")
    parser.add_argument("kernel", help="Kernel class name, for example AddKernel.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--max-tool-rounds", type=int, default=20)
    parser.add_argument("--reasoning-effort", default="high")
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
            reasoning_effort=(
                None if args.reasoning_effort == "none" else args.reasoning_effort
            ),
            trace_path=args.trace_path,
            start_json=args.start_json,
        )
    )


if __name__ == "__main__":
    main()
