import json
import re
from datetime import datetime
from pathlib import Path

from .evaluation import json_default


def create_trace_directory(trace_root, kernel_name, model, reasoning_effort):
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    safe_values = [
        re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
        for value in (kernel_name, model, reasoning_effort)
    ]
    trace_directory = Path(trace_root) / f"{timestamp}_{'_'.join(safe_values)}"
    trace_directory.mkdir(parents=True, exist_ok=False)
    return trace_directory


def write_trace(trace_path, events):
    if trace_path is not None:
        (trace_path / "events_speedup_vs_triton_pending.json").write_text(
            json.dumps(events, indent=2, default=json_default),
            encoding="utf-8",
        )


def _artifact_stem(
    *,
    prompt_name,
    round_index,
    attempt_index,
    speedup_vs_triton,
    candidate_index=None,
):
    candidate_suffix = (
        "" if candidate_index is None else f"_candidate_{candidate_index:02d}"
    )
    speedup = "pending" if speedup_vs_triton is None else f"{speedup_vs_triton:.4f}x"
    return (
        f"iteration_{round_index:03d}{candidate_suffix}_try_{attempt_index:02d}_"
        f"{prompt_name}_speedup_vs_triton_{speedup}"
    )


def record_prompt(
    responses,
    trace_path,
    *,
    prompt_name,
    prompt,
    round_index,
    attempt_index=0,
    candidate_index=None,
    idea=None,
    speedup_vs_triton=None,
):
    artifact_stem = _artifact_stem(
        prompt_name=prompt_name,
        round_index=round_index,
        attempt_index=attempt_index,
        candidate_index=candidate_index,
        speedup_vs_triton=speedup_vs_triton,
    )
    (trace_path / f"{artifact_stem}_prompt.txt").write_text(
        prompt,
        encoding="utf-8",
    )
    responses.append(
        {
            "type": "prompt",
            "prompt_name": prompt_name,
            "round_index": round_index,
            "attempt_index": attempt_index,
            "candidate_index": candidate_index,
            "idea": idea,
            "prompt": prompt,
        }
    )
    write_trace(trace_path, responses)


def record_generated_json(
    trace_path,
    generated_value,
    *,
    prompt_name,
    round_index,
    attempt_index,
    speedup_vs_triton,
    candidate_index=None,
):
    artifact_stem = _artifact_stem(
        prompt_name=prompt_name,
        round_index=round_index,
        attempt_index=attempt_index,
        candidate_index=candidate_index,
        speedup_vs_triton=speedup_vs_triton,
    )
    (trace_path / f"{artifact_stem}.json").write_text(
        json.dumps(generated_value, indent=2, default=json_default),
        encoding="utf-8",
    )


def record_generated_candidate(
    trace_path,
    candidate,
    evaluation,
    *,
    prompt_name,
    round_index,
    attempt_index,
    candidate_index=None,
):
    artifact_stem = _artifact_stem(
        prompt_name=prompt_name,
        round_index=round_index,
        attempt_index=attempt_index,
        candidate_index=candidate_index,
        speedup_vs_triton=evaluation.speedup_vs_triton,
    )
    candidate_json = candidate.model_dump_json(exclude_none=False, indent=2)
    payload = {
        "iteration": round_index,
        "try": attempt_index,
        "candidate_index": candidate_index,
        "prompt_name": prompt_name,
        "candidate": json.loads(candidate_json),
        "evaluation": evaluation.to_dict(),
    }
    (trace_path / f"{artifact_stem}.json").write_text(
        json.dumps(payload, indent=2, default=json_default),
        encoding="utf-8",
    )
    (trace_path / f"{artifact_stem}.ptx").write_text(
        candidate.ptx.rstrip() + "\n",
        encoding="utf-8",
    )
