import json
import math
import re
from csv import reader
from dataclasses import asdict
from datetime import datetime
from functools import lru_cache
from io import StringIO
from pathlib import Path

import torch

from .cost import read_cost_total
from .evaluation import json_default, normalize_nested_json

NCU_KERNEL_NAME = "kernel"
STALL_COLUMNS = {
    "long_scoreboard": "stall_long_sb",
    "short_scoreboard": "stall_short_sb",
    "math_pipe": "stall_math",
    "barrier": "stall_barrier",
}
EXCLUDED_STALL_COLUMNS = {"stall_selected"}
TOP_STALL_LINES = 20


@lru_cache(maxsize=1)
def gpu_type():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(torch.cuda.current_device())


def autotune_metrics(operator):
    tuning_result = getattr(operator, "tuning_result", None)
    if tuning_result is not None and not isinstance(tuning_result, dict):
        tuning_result = asdict(tuning_result)
    return {
        "operator": type(operator).__name__,
        "constexpr_values": dict(getattr(operator, "constexpr_values", {})),
        "selected_config": dict(getattr(operator, "best_config", {})),
        "tuning_result": tuning_result,
    }


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
            json.dumps(
                normalize_nested_json(
                    [{**event, "gpu_type": gpu_type()} for event in events]
                ),
                indent=2,
                default=json_default,
            ),
            encoding="utf-8",
        )


def _run_cost_usd_so_far(trace_path):
    return read_cost_total(Path(trace_path) / "prices.log")


def write_line_by_line_ncu_report(trace_path, candidate, evaluation):
    ncu_report = evaluation.ncu_report
    source_report = ncu_report.get("source_report", "")
    if not source_report:
        return

    (trace_path / "ncu_source_report.csv").write_text(
        source_report,
        encoding="utf-8",
    )
    annotated_report = render_annotated_ptx_report(candidate.ptx, ncu_report)
    (trace_path / "NCU_report_line_by_line.md").write_text(
        annotated_report,
        encoding="utf-8",
    )


def render_annotated_ptx_report(ptx, ncu_report):
    source_report = ncu_report.get("source_report", "")
    lines = [
        "# NCU Report, Annotated PTX",
        "",
        "Only the candidate entry named `kernel` is included. Metrics are collected "
        "at SASS PCs and grouped under the PTX line carried by profiling-only "
        "`.loc` metadata.",
        "",
        "```ptx",
    ]
    ptx_lines = ptx.splitlines()
    if not source_report:
        lines.extend(
            [
                "// NCU line annotations are unavailable because the source report "
                "was not collected.",
                *ptx_lines,
                "```",
                "",
            ]
        )
        return "\n".join(lines)

    rows_by_line = _correlate_source_rows(
        ptx_lines,
        _read_ncu_source_rows(source_report),
    )
    records = [
        _ptx_line_record(line_number, ptx_line, rows_by_line.get(line_number, []))
        for line_number, ptx_line in enumerate(ptx_lines, start=1)
    ]
    top_records = _top_stall_records(records)
    rank_by_line_number = {
        record["line_number"]: rank
        for rank, record in enumerate(top_records, start=1)
    }
    lines.insert(
        4,
        f"The {len(top_records)} PTX lines with the most total stall samples are "
        "annotated below.",
    )
    for record in records:
        lines.append(
            _format_annotated_ptx_line(
                record,
                rank=rank_by_line_number.get(record["line_number"]),
            )
        )
    lines.extend(["```", ""])
    return "\n".join(lines)


def ncu_instruction_issues(ptx, ncu_report):
    """Return compact NCU diagnostics for the PTX lines shown to the model."""
    source_report = ncu_report.get("source_report", "")
    if not source_report:
        return []

    ptx_lines = ptx.splitlines()
    rows_by_line = _correlate_source_rows(
        ptx_lines,
        _read_ncu_source_rows(source_report),
    )
    records = [
        _ptx_line_record(line_number, ptx_line, rows_by_line.get(line_number, []))
        for line_number, ptx_line in enumerate(ptx_lines, start=1)
    ]
    top_records = _top_stall_records(records)
    return [
        {
            "rank": rank,
            "instruction": record["ptx"].strip(),
            "lines": [record["line_number"]],
            "total_stall_samples": _total_stall_samples(record),
            "stall_breakdown": {
                name: value for name, value in record["stalls"].items() if value
            },
        }
        for rank, record in enumerate(top_records, start=1)
    ]


def _read_ncu_source_rows(source_report):
    csv_rows = list(reader(StringIO(source_report)))
    candidate_rows = []
    row_index = 0
    while row_index < len(csv_rows):
        row = csv_rows[row_index]
        if len(row) < 2 or row[0] not in {"Function Name", "Kernel Name"}:
            row_index += 1
            continue
        kernel_name = row[1]
        row_index += 1
        if row_index >= len(csv_rows):
            break
        columns = _source_columns(csv_rows[row_index])
        row_index += 1
        block_headers = {"File Path", "Function Name", "Kernel Name"}
        while row_index < len(csv_rows) and (
            not csv_rows[row_index] or csv_rows[row_index][0] not in block_headers
        ):
            values = csv_rows[row_index]
            if kernel_name == NCU_KERNEL_NAME and values:
                candidate_rows.append(dict(zip(columns, values, strict=False)))
            row_index += 1
    return candidate_rows


def _source_columns(columns):
    source_index = 0
    normalized_columns = []
    for column in columns:
        if column != "Source":
            normalized_columns.append(column)
            continue
        source_index += 1
        normalized_columns.append("PTX Source" if source_index == 1 else "SASS Source")
    return normalized_columns


def _correlate_source_rows(ptx_lines, source_rows):
    rows_by_line = {}
    current_line = None
    search_start = 0
    for row in source_rows:
        address = row.get("Address", "").strip()
        line_number = row.get("Line No", "").strip()
        if line_number.isdigit():
            current_line = int(line_number)
            search_start = current_line
            continue
        if address in {"", "-", "..."}:
            source = row.get("PTX Source", "").strip()
            matched_line = _match_ptx_source_line(ptx_lines, source, search_start)
            if matched_line is not None:
                current_line = matched_line
                search_start = matched_line
            continue
        if current_line is not None:
            rows_by_line.setdefault(current_line, []).append(row)
    return rows_by_line


def _match_ptx_source_line(ptx_lines, source, search_start):
    source = re.sub(r"^\s*\d+\s+", "", source).strip()
    if not source:
        return None
    for offset in range(search_start, len(ptx_lines)):
        if ptx_lines[offset].strip() == source:
            return offset + 1
    return None


def _ptx_line_record(line_number, ptx_line, sass_rows):
    executable = _is_executable_ptx_line(ptx_line)
    if not sass_rows:
        return {
            "line_number": line_number,
            "ptx": ptx_line,
            "executable": executable,
            "sass": [],
            "issues": [],
        }

    sample_count = _not_issued_sample_count(sass_rows)
    stalls = _stall_reasons(sass_rows)
    global_sectors = _sum_column(sass_rows, "L2 Theoretical Sectors Global")
    ideal_global_sectors = _sum_column(
        sass_rows,
        "L2 Theoretical Sectors Global Ideal",
    )
    shared_wavefronts = _sum_column(sass_rows, "L1 Wavefronts Shared")
    ideal_shared_wavefronts = _sum_column(
        sass_rows,
        "L1 Wavefronts Shared Ideal",
    )
    issues = []
    for name, value in stalls.items():
        percentage = value / sample_count * 100 if sample_count else 0
        if percentage >= 20:
            issues.append(f"{name} stalls {percentage:.1f}% ({value:g} samples)")
    for label, actual, ideal in (
        ("global sectors", global_sectors, ideal_global_sectors),
        ("shared wavefronts", shared_wavefronts, ideal_shared_wavefronts),
    ):
        if ideal and actual / ideal >= 2:
            issues.append(f"{label} {actual / ideal:.2f}x ideal ({actual:g}/{ideal:g})")

    dominant_scoreboard = max(
        ("long", stalls["long_scoreboard"]),
        ("short", stalls["short_scoreboard"]),
        key=lambda item: item[1],
    )
    return {
        "line_number": line_number,
        "ptx": ptx_line,
        "executable": executable,
        "sass": [
            {
                "pc": row.get("Address", ""),
                "opcode": row.get("SASS Source", "").strip(),
            }
            for row in sass_rows
        ],
        "executed_count": _sum_column(sass_rows, "Instructions Executed"),
        "sample_count": sample_count,
        "stalls": stalls,
        "thread_instructions": _sum_column(
            sass_rows,
            "Thread Instructions Executed",
        ),
        "predicated_on_thread_instructions": _sum_column(
            sass_rows,
            "Predicated-On Thread Instructions Executed",
        ),
        "memory": _memory_metrics(sass_rows),
        "global_sectors": global_sectors,
        "ideal_global_sectors": ideal_global_sectors,
        "shared_wavefronts": shared_wavefronts,
        "ideal_shared_wavefronts": ideal_shared_wavefronts,
        "scoreboard": (
            f"{dominant_scoreboard[0]} ({dominant_scoreboard[1]:g} samples)"
            if dominant_scoreboard[1]
            else "none sampled"
        ),
        "issues": issues,
    }


def _is_executable_ptx_line(ptx_line):
    stripped = ptx_line.strip()
    return bool(stripped.endswith(";") and not stripped.startswith((".", "//")))


def _sum_column(rows, column):
    return sum(_numeric_value(row.get(column, "")) for row in rows)


def _stall_reasons(rows):
    columns = {column for row in rows for column in row}
    not_issued_columns = {
        column.removesuffix(" (Not Issued)"): column
        for column in columns
        if column.endswith(" (Not Issued)")
    }
    stalls = {
        name: _sum_column(rows, not_issued_columns.get(column, column))
        for name, column in STALL_COLUMNS.items()
    }
    known_columns = set(STALL_COLUMNS.values())
    selected_columns = not_issued_columns or {column: column for column in columns}
    for base_column, column in sorted(selected_columns.items()):
        if (
            not base_column.startswith("stall_")
            or base_column in known_columns
            or base_column in EXCLUDED_STALL_COLUMNS
        ):
            continue
        stalls[base_column.removeprefix("stall_")] = _sum_column(rows, column)
    return stalls


def _not_issued_sample_count(rows):
    for column in ("Warp Stall Sampling (Not-issued Samples)", "# Samples"):
        if any(column in row for row in rows):
            return _sum_column(rows, column)
    return 0.0


def _memory_metrics(rows):
    memory_space = _first_text_value(rows, "Address Space")
    if memory_space is None:
        return None

    memory_space = re.sub(r"\(\d+\)$", "", memory_space).lower()
    operation = _first_text_value(rows, "Access Operation")
    access_size = _first_text_value(rows, "Access Size")
    if operation is not None:
        operation = re.sub(r"\(\d+\)$", "", operation).lower()
    if access_size is not None:
        access_size = re.sub(r"\(\d+\)$", "", access_size)

    if memory_space == "shared":
        actual = _sum_column(rows, "L1 Wavefronts Shared")
        ideal = _sum_column(rows, "L1 Wavefronts Shared Ideal")
        excessive = _sum_column(rows, "L1 Wavefronts Shared Excessive")
        return {
            "space": memory_space,
            "operation": operation,
            "access_size": access_size,
            "unit": "wavefronts",
            "actual": actual,
            "ideal": ideal,
            "excessive": excessive,
            "bank_conflicts": _sum_column(rows, "L1 Conflicts Shared N-Way"),
        }
    if memory_space in {"global", "local"}:
        actual = _sum_column(rows, "L2 Theoretical Sectors Global")
        ideal = _sum_column(rows, "L2 Theoretical Sectors Global Ideal")
        excessive = _sum_column(rows, "L2 Theoretical Sectors Global Excessive")
        return {
            "space": memory_space,
            "operation": operation,
            "access_size": access_size,
            "unit": "sectors",
            "actual": actual,
            "ideal": ideal,
            "excessive": excessive,
        }
    return None


def _first_text_value(rows, column):
    for row in rows:
        value = row.get(column, "").strip()
        if value and value != "-":
            return value
    return None


def _numeric_value(value):
    try:
        return float(value.replace(",", "")) if value not in ("", "-") else 0.0
    except ValueError:
        return 0.0


def _mark_frequent_lines(records):
    executed_counts = sorted(
        record["executed_count"] for record in records if "executed_count" in record
    )
    if not executed_counts:
        return
    threshold = executed_counts[int((len(executed_counts) - 1) * 0.95)]
    if not threshold:
        return
    for record in records:
        executed_count = record.get("executed_count", 0)
        if executed_count >= threshold:
            record["issues"].append(
                f"high execution frequency ({executed_count:g}, >= p95 {threshold:g})"
            )


def _format_annotated_ptx_line(record, *, rank=None):
    line = record["ptx"]
    if rank is None:
        return line

    sample_count = record["sample_count"]
    reasons = [
        (
            f"{name}: {value:g} ({value / sample_count * 100:.1f}%)"
            if sample_count
            else f"{name}: {value:g}"
        )
        for name, value in record["stalls"].items()
        if value
    ]
    comment_parts = [
        f"NCU stall rank: {rank}",
        f"total_stall_samples: {_total_stall_samples(record):g}",
        f"stalls_not_issued: {sample_count:g}",
    ]
    if reasons:
        comment_parts.append(", ".join(reasons))

    executed_count = record["executed_count"]
    if executed_count:
        avg_threads_executed = record["thread_instructions"] / executed_count
        avg_threads_predicated_on = (
            record["predicated_on_thread_instructions"] / executed_count
        )
        if avg_threads_predicated_on < avg_threads_executed:
            comment_parts.append(
                "threads: "
                f"executed {avg_threads_executed:.1f}, "
                f"predicated_on {avg_threads_predicated_on:.1f}"
            )

    memory = record["memory"]
    if memory is not None:
        access = " ".join(
            value
            for value in (
                memory["space"],
                memory["operation"],
                memory["access_size"],
            )
            if value is not None
        )
        memory_description = (
            f"memory: {access}; {memory['unit']}: {memory['actual']:g}/"
            f"{memory['ideal']:g}; excessive: {memory['excessive']:g}"
        )
        if memory["ideal"]:
            memory_description += (
                f"; overhead: {memory['actual'] / memory['ideal']:.2f}x"
            )
        if "bank_conflicts" in memory:
            memory_description += f"; bank_conflicts: {memory['bank_conflicts']:g}"
        comment_parts.append(memory_description)
    return line + " // " + "; ".join(comment_parts)


def _top_stall_records(records):
    ranked_records = [
        record
        for record in records
        if (
            record["executable"]
            and record["sass"]
            and _total_stall_samples(record) > 0
        )
    ]
    return sorted(
        ranked_records,
        key=lambda record: (-_total_stall_samples(record), record["line_number"]),
    )[:TOP_STALL_LINES]


def _total_stall_samples(record):
    return sum(record["stalls"].values())


def _problem_summary(records):
    problematic = [record for record in records if record["issues"]]
    problematic.sort(
        key=lambda record: (
            record.get("sample_count", 0),
            record.get("executed_count", 0),
        ),
        reverse=True,
    )
    if not problematic:
        return ["- No problematic correlated PTX line crossed the thresholds."]
    return [
        f"- PTX line {record['line_number']}: " + "; ".join(record["issues"])
        for record in problematic[:20]
    ]


def record_tool_call(
    trace_path,
    *,
    provider,
    tool_name,
    call_id,
    answer,
    resulting_payload=None,
    verified_ptx=None,
    speedup_vs_triton=None,
    autotune_metrics=None,
):
    tool_output_path = Path(trace_path) / "tool_output"
    tool_output_path.mkdir(parents=True, exist_ok=True)
    safe_tool_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(tool_name)).strip("._")
    artifact_index = sum(1 for _ in tool_output_path.glob("*.json")) + 1
    artifact_name = f"{artifact_index:03d}_{safe_tool_name or 'tool'}.json"
    record = {
        "provider": provider,
        "tool_name": tool_name,
        "call_id": call_id,
        "answer": normalize_nested_json(answer),
        "run_cost_usd_so_far": _run_cost_usd_so_far(trace_path),
    }
    if autotune_metrics is not None:
        record["autotune_metrics"] = normalize_nested_json(autotune_metrics)
    successful = verified_ptx is not None
    if successful and speedup_vs_triton is not None:
        record["speedup_vs_triton"] = speedup_vs_triton
    if resulting_payload is not None:
        record["resulting_payload"] = normalize_nested_json(resulting_payload)
    if successful:
        record["ptx_hyperparameters"] = {
            name: resulting_payload.get(name)
            for name in ("num_threads_x", "num_threads_y", "num_threads_z")
            if resulting_payload is not None and resulting_payload.get(name) is not None
        }
    (tool_output_path / artifact_name).write_text(
        json.dumps(
            record,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )
    if successful:
        speedup_suffix = "unverified"
        if isinstance(speedup_vs_triton, (int, float)) and math.isfinite(
            speedup_vs_triton
        ):
            speedup_suffix = f"speedup_vs_triton_{speedup_vs_triton:.4f}x"
        ptx_path = tool_output_path / (
            f"{artifact_index:03d}_{safe_tool_name or 'tool'}_{speedup_suffix}.ptx"
        )
        ptx_path.write_text(
            verified_ptx,
            encoding="utf-8",
        )


def _artifact_stem(
    *,
    prompt_name,
    round_index,
    attempt_index,
    speedup_vs_triton,
    candidate_index=None,
    include_speedup=True,
):
    candidate_suffix = (
        "" if candidate_index is None else f"_candidate_{candidate_index:02d}"
    )
    artifact_stem = (
        f"iteration_{round_index:03d}{candidate_suffix}_try_{attempt_index:02d}_"
        f"{prompt_name}"
    )
    if not include_speedup:
        return artifact_stem

    speedup = "pending" if speedup_vs_triton is None else f"{speedup_vs_triton:.4f}x"
    return f"{artifact_stem}_speedup_vs_triton_{speedup}"


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
        include_speedup=False,
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
    include_speedup=True,
    autotune_metrics=None,
):
    artifact_stem = _artifact_stem(
        prompt_name=prompt_name,
        round_index=round_index,
        attempt_index=attempt_index,
        candidate_index=candidate_index,
        speedup_vs_triton=speedup_vs_triton,
        include_speedup=include_speedup,
    )
    (trace_path / f"{artifact_stem}.json").write_text(
        json.dumps(
            normalize_nested_json(
                {
                    **generated_value,
                    "autotune_metrics": autotune_metrics,
                    "gpu_type": gpu_type(),
                    "run_cost_usd_so_far": _run_cost_usd_so_far(trace_path),
                }
            ),
            indent=2,
            default=json_default,
        ),
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
    autotune_metrics=None,
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
        "evaluation": json.loads(evaluation.to_json(indent=2)),
        "autotune_metrics": autotune_metrics,
        "gpu_type": gpu_type(),
        "run_cost_usd_so_far": _run_cost_usd_so_far(trace_path),
    }
    (trace_path / f"{artifact_stem}.json").write_text(
        json.dumps(
            normalize_nested_json(payload),
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )
    (trace_path / f"{artifact_stem}.ptx").write_text(
        candidate.ptx.rstrip() + "\n",
        encoding="utf-8",
    )
