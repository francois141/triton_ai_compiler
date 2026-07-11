import json

from utils.evaluation import json_default


def write_trace(trace_path, events):
    if trace_path is not None:
        trace_path.write_text(
            json.dumps(events, indent=2, default=json_default),
            encoding="utf-8",
        )


def final_path(trace_path):
    return trace_path.with_name(f"{trace_path.stem}_final.json")


def record_prompt(
    responses,
    trace_path,
    *,
    prompt_name,
    prompt,
    round_index,
    candidate_index=None,
    idea=None,
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
    write_trace(trace_path, responses)
