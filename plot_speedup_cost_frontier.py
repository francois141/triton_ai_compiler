import argparse
import ast
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import tiktoken
from matplotlib.lines import Line2D

from clean_ptx import clean_ptx

plt.switch_backend("Agg")


LOGGER = logging.getLogger(__name__)
GRID_PLOT_BASENAME = "speedup_cost_frontiers_grid"
TOKEN_EXPANSION_PLOT_BASENAME = "ptx_token_expansion_factors"
GEMM_LLM_FRONTIER_PLOT_BASENAME = "gemm_llm_cost_frontiers"
GEMM_GPU_TYPE = "NVIDIA L40S"
GEMM_COST_LIMIT_USD = 15.0
# Restrict this set of figures to GPT-6 experiment traces.
MODEL_PREFIXES = ("gpt-6",)
GEMM_FAILED_MODELS_BY_PROVIDER = {
    "OpenAI LLMs": ("gpt-5.6-luna",),
    "Anthropic LLMs": (
        "claude-haiku-4-5-20251001",
        "claude-opus-5",
    ),
}
# These historical GEMM traces predate persisted GPU metadata, but were all
# collected on the L40S used for the GEMM frontier.
LEGACY_L40S_GEMM_MODELS = {
    "gpt-5.6-sol",
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-sonnet-5",
}
CORRECTION_FACTOR_FILENAME = "correction factor.txt"
PENDING_SPEEDUP_EVENTS_FILENAMES = (
    "events_speedup_vs_triton_pending.json",
    # Retain compatibility with traces created under the historical filename.
    "events_speedup_vst_triton_pending.json",
)
PLOT_FORMATS = ("jpeg", "pdf")
GRID_ROWS = 3
GRID_COLUMNS = 4
GRID_SIZE = GRID_ROWS * GRID_COLUMNS
AXIS_TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
GPU_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
FLOAT_PRECISION_LABELS = {
    "Float16": "Floating Point 16",
    "Float8": "Floating Point 8",
}
RUN_DIRECTORY_PATTERN = re.compile(r"^\d{12}_")
REASONING_EFFORT_PATTERN = re.compile(
    r"_(?:low|medium|high|max|xhigh|ultra|none)$", re.IGNORECASE
)
KERNEL_SOURCE_DIRECTORY = (
    Path(__file__).resolve().parent / "triton_ptx" / "triton_ptx" / "kernels"
)
TOKEN_ENCODING = tiktoken.get_encoding("o200k_base")


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _accepted_speedup(data, json_path):
    evaluation = data.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("passed") is True:
        return _finite_number(evaluation.get("speedup_vs_triton"))

    speedup = _finite_number(data.get("speedup_vs_triton"))
    if speedup is not None and json_path.parent.name == "tool_output":
        return speedup

    if json_path.name.startswith("final_speedup_vs_triton_"):
        return _finite_number(data.get("speedup"))
    return None


def _floating_point_precision(kernel_name):
    for precision, label in FLOAT_PRECISION_LABELS.items():
        if precision in kernel_name:
            return label
    return None


def _kernel_name_from_path(trace_directory, relative_path):
    run_directory = (
        trace_directory.name
        if relative_path.parts[0] == "tool_output"
        else relative_path.parts[0]
    )
    name_parts = run_directory.split("_", maxsplit=2)
    if len(name_parts) >= 2 and name_parts[0].isdigit():
        return name_parts[1]
    return run_directory


def _run_directory(relative_path):
    for index, path_part in enumerate(relative_path.parts):
        if RUN_DIRECTORY_PATTERN.match(path_part):
            return Path(*relative_path.parts[: index + 1])
    return Path(relative_path.parts[0])


def _model_from_run_directory(run_directory):
    name_parts = run_directory.name.split("_", maxsplit=2)
    if len(name_parts) != 3:
        return "Unknown model"
    return REASONING_EFFORT_PATTERN.sub("", name_parts[2])


def _is_included_run(run_directory):
    return _model_from_run_directory(run_directory).startswith(MODEL_PREFIXES)


def _provider_group(provider, model):
    provider = provider.lower() if isinstance(provider, str) else ""
    model = model.lower()
    if provider == "openai" or model.startswith(("gpt-", "o1", "o3", "o4")):
        return "OpenAI LLMs"
    if provider == "anthropic" or model.startswith("claude-"):
        return "Anthropic LLMs"
    return "Other LLMs"


def _is_gemm_kernel(kernel_name):
    normalized_name = kernel_name.lower()
    # The LLM frontier compares the common base GEMM workload.  In particular,
    # do not combine it with fused GEMM operators (for example, GEMM + GELU),
    # which have a different Triton baseline and cannot share a frontier.
    return "matrixmultiplication" in normalized_name


def _is_gemm_gpu(gpu_type):
    return gpu_type.casefold() == GEMM_GPU_TYPE.casefold()


def _is_l40s_gemm_point(point):
    if _is_gemm_gpu(point["gpu_type"]):
        return True
    # This trace predates persisted GPU metadata but was run on the L40S.
    return (
        point["gpu_type"] == "Unknown GPU" and point["model"] in LEGACY_L40S_GEMM_MODELS
    )


def _correction_factor(run_directory):
    correction_path = run_directory / CORRECTION_FACTOR_FILENAME
    try:
        correction_factor = float(correction_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 1.0
    if math.isfinite(correction_factor) and correction_factor > 0:
        return correction_factor
    LOGGER.warning("Ignoring invalid correction factor in %s", correction_path)
    return 1.0


def _kernel_name(data, trace_directory, relative_path):
    evaluation = data.get("evaluation")
    kernel_name = (
        evaluation.get("kernel_name") if isinstance(evaluation, dict) else None
    )
    autotune_metrics = data.get("autotune_metrics")
    if kernel_name is None and isinstance(autotune_metrics, dict):
        kernel_name = autotune_metrics.get("operator")
    return kernel_name or _kernel_name_from_path(trace_directory, relative_path)


def _gpu_type(data):
    gpu_type = data.get("gpu_type")
    if isinstance(gpu_type, str) and gpu_type:
        return gpu_type
    evaluation = data.get("evaluation")
    ncu_report = data.get("ncu_report")
    if ncu_report is None and isinstance(evaluation, dict):
        ncu_report = evaluation.get("ncu_report")
    summary = ncu_report.get("summary") if isinstance(ncu_report, dict) else None
    hardware = summary.get("hardware") if isinstance(summary, dict) else None
    display_name = hardware.get("display_name") if isinstance(hardware, dict) else None
    return display_name if isinstance(display_name, str) else None


def _pending_events_gpu_type(data):
    """Return the GPU recorded with an in-progress run's trace events."""
    if not isinstance(data, list):
        return None
    for event in data:
        if not isinstance(event, dict):
            continue
        gpu_type = _gpu_type(event)
        if gpu_type is not None:
            return gpu_type
    return None


def _add_prior_run_costs(
    accepted_kernels,
    run_costs,
    gpu_types_by_run,
    models_by_run,
):
    points_by_kernel_gpu_model_and_run = defaultdict(lambda: defaultdict(list))
    run_costs_by_kernel_gpu_and_model = defaultdict(dict)
    for point in accepted_kernels:
        gpu_type = gpu_types_by_run.get(point["run_directory"], "Unknown GPU")
        model = models_by_run[point["run_directory"]]
        point["gpu_type"] = gpu_type
        points_by_kernel_gpu_model_and_run[(point["kernel"], gpu_type, model)][
            point["run_directory"]
        ].append(point)

    for (kernel_name, run_directory), cost in run_costs.items():
        gpu_type = gpu_types_by_run.get(run_directory, "Unknown GPU")
        model = models_by_run[run_directory]
        run_costs_by_kernel_gpu_and_model[(kernel_name, gpu_type, model)][
            run_directory
        ] = cost

    for (
        kernel_name,
        gpu_type,
        model,
    ), costs_by_run in run_costs_by_kernel_gpu_and_model.items():
        prior_cost = 0.0
        for run_directory in sorted(costs_by_run):
            for point in points_by_kernel_gpu_model_and_run[
                (kernel_name, gpu_type, model)
            ][run_directory]:
                point["cost_usd"] += prior_cost
            prior_cost += costs_by_run[run_directory]


def load_accepted_kernels(trace_directory):
    accepted_kernels = []
    run_costs = defaultdict(float)
    gpu_types_by_run = {}
    pending_gpu_types_by_run = {}
    providers_by_run = {}
    for json_path in trace_directory.rglob("*.json"):
        relative_path = json_path.relative_to(trace_directory)
        try:
            with json_path.open(encoding="utf-8") as json_file:
                data = json.load(json_file)
        except (json.JSONDecodeError, OSError) as error:
            LOGGER.warning("Skipping unreadable JSON %s: %s", json_path, error)
            continue
        run_directory = _run_directory(relative_path)
        if not _is_included_run(run_directory):
            continue
        if json_path.name in PENDING_SPEEDUP_EVENTS_FILENAMES:
            gpu_type = _pending_events_gpu_type(data)
            if gpu_type is not None:
                pending_gpu_types_by_run[run_directory] = gpu_type
            continue
        if not isinstance(data, dict):
            continue

        gpu_type = _gpu_type(data)
        if gpu_type is not None:
            gpu_types_by_run[run_directory] = gpu_type
        provider = data.get("provider")
        if isinstance(provider, str) and provider:
            providers_by_run[run_directory] = provider

        cost = _finite_number(data.get("run_cost_usd_so_far"))
        if cost is None or cost < 0:
            continue

        kernel_name = _kernel_name(data, trace_directory, relative_path)
        precision = _floating_point_precision(kernel_name)
        if precision is None:
            continue
        run_costs[kernel_name, run_directory] = max(
            run_costs[kernel_name, run_directory], cost
        )

        speedup = _accepted_speedup(data, json_path)
        if speedup is None:
            continue
        accepted_kernels.append(
            {
                "cost_usd": cost,
                "speedup_vs_triton": speedup * _correction_factor(run_directory),
                "kernel": kernel_name,
                "precision": precision,
                "source": str(relative_path),
                "run_directory": run_directory,
                "model": _model_from_run_directory(run_directory),
            }
        )
    for run_directory, gpu_type in pending_gpu_types_by_run.items():
        gpu_types_by_run.setdefault(run_directory, gpu_type)
    models_by_run = {
        run_directory: _model_from_run_directory(run_directory)
        for _, run_directory in run_costs
    }
    _add_prior_run_costs(
        accepted_kernels,
        run_costs,
        gpu_types_by_run,
        models_by_run,
    )
    for point in accepted_kernels:
        point["provider"] = _provider_group(
            providers_by_run.get(point["run_directory"]), point["model"]
        )
        point.pop("run_directory")
    return accepted_kernels


def pareto_frontier(points):
    frontier = []
    best_speedup = -math.inf
    for point in sorted(
        points,
        key=lambda item: (item["cost_usd"], -item["speedup_vs_triton"]),
    ):
        if point["speedup_vs_triton"] > best_speedup:
            frontier.append(point)
            best_speedup = point["speedup_vs_triton"]
    return frontier


def _display_kernel_name(kernel_name):
    for precision in FLOAT_PRECISION_LABELS:
        kernel_name = kernel_name.replace(precision, "")
    return kernel_name


def _triton_source_path(kernel_name):
    class_names = [kernel_name]
    if not kernel_name.endswith("Kernel"):
        class_names.append(f"{kernel_name}Kernel")
    class_patterns = [
        re.compile(rf"^class {re.escape(class_name)}\b", re.MULTILINE)
        for class_name in class_names
    ]
    for source_path in KERNEL_SOURCE_DIRECTORY.rglob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        if any(pattern.search(source) for pattern in class_patterns):
            return source_path
    return None


def _triton_kernel_source(kernel_name):
    source_path = _triton_source_path(kernel_name)
    if source_path is None:
        LOGGER.warning("No Triton source was found for %s.", kernel_name)
        return None

    source = source_path.read_text(encoding="utf-8")
    source_tree = ast.parse(source, filename=source_path)
    class_names = {kernel_name, f"{kernel_name}Kernel"}
    for node in source_tree.body:
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            return ast.get_source_segment(source, node)
    LOGGER.warning("No Triton kernel class was found in %s.", source_path)


def _token_count(text):
    return len(TOKEN_ENCODING.encode(text))


def load_gpu_types_by_run(trace_directory):
    gpu_types_by_run = {}
    pending_gpu_types_by_run = {}
    for json_path in trace_directory.rglob("*.json"):
        relative_path = json_path.relative_to(trace_directory)
        try:
            with json_path.open(encoding="utf-8") as json_file:
                data = json.load(json_file)
        except (json.JSONDecodeError, OSError) as error:
            LOGGER.warning("Skipping unreadable JSON %s: %s", json_path, error)
            continue
        run_directory = _run_directory(relative_path)
        if json_path.name in PENDING_SPEEDUP_EVENTS_FILENAMES:
            gpu_type = _pending_events_gpu_type(data)
            if gpu_type is not None:
                pending_gpu_types_by_run[run_directory] = gpu_type
            continue
        if not isinstance(data, dict):
            continue
        gpu_type = _gpu_type(data)
        if gpu_type is not None:
            gpu_types_by_run[run_directory] = gpu_type
    for run_directory, gpu_type in pending_gpu_types_by_run.items():
        gpu_types_by_run.setdefault(run_directory, gpu_type)
    return gpu_types_by_run


def _llm_ptx_path(run_directory):
    final_path = run_directory / "final_candidate.ptx"
    if final_path.is_file():
        return final_path

    best_path = None
    best_speedup = -math.inf
    for ptx_path in sorted(run_directory.rglob("*speedup_vs_triton_*.ptx")):
        json_path = ptx_path.with_suffix(".json")
        if ptx_path.parent.name == "tool_output":
            json_path = ptx_path.with_name(
                ptx_path.name.rsplit("_speedup_vs_triton_", 1)[0] + ".json"
            )
        try:
            with json_path.open(encoding="utf-8") as json_file:
                data = json.load(json_file)
        except (json.JSONDecodeError, OSError) as error:
            LOGGER.warning("Skipping unreadable JSON %s: %s", json_path, error)
            continue
        if not isinstance(data, dict):
            continue
        speedup = _accepted_speedup(data, json_path)
        if speedup is not None and speedup > best_speedup:
            best_path = ptx_path
            best_speedup = speedup
    if best_path is not None:
        LOGGER.warning("No final LLM PTX; using best verified candidate %s.", best_path)
    return best_path


def load_token_expansion_data(trace_directory):
    token_counts = []
    gpu_types_by_run = load_gpu_types_by_run(trace_directory)
    for ptx_path in sorted(trace_directory.rglob("triton_generated.ptx")):
        relative_path = ptx_path.relative_to(trace_directory)
        run_directory = _run_directory(relative_path)
        if not _is_included_run(run_directory):
            continue
        kernel_name = _kernel_name_from_path(trace_directory, run_directory)
        if "Float16" not in kernel_name:
            continue
        triton_source = _triton_kernel_source(kernel_name)
        if triton_source is None:
            continue
        llm_ptx_path = _llm_ptx_path(ptx_path.parent)
        if llm_ptx_path is None:
            LOGGER.warning("No verified LLM PTX was found for %s.", kernel_name)
            continue
        triton_generated_ptx = clean_ptx(ptx_path.read_text(encoding="utf-8"))
        llm_generated_ptx = llm_ptx_path.read_text(encoding="utf-8")
        triton_tokens = _token_count(triton_source)
        if triton_tokens == 0:
            LOGGER.warning("Triton source for %s has no tokens.", kernel_name)
            continue
        triton_generated_ptx_tokens = _token_count(triton_generated_ptx)
        llm_generated_ptx_tokens = _token_count(llm_generated_ptx)
        token_counts.append(
            {
                "kernel": _display_kernel_name(kernel_name),
                "gpu_type": gpu_types_by_run.get(run_directory, "Unknown GPU"),
                "triton_generated_ptx_tokens": triton_generated_ptx_tokens,
                "llm_generated_ptx_tokens": llm_generated_ptx_tokens,
                "triton_generated_expansion_factor": (
                    triton_generated_ptx_tokens / triton_tokens
                ),
                "llm_generated_expansion_factor": (
                    llm_generated_ptx_tokens / triton_tokens
                ),
            }
        )
    return token_counts


def _plot_frontier(axis, frontier, color, gpu_type):
    costs = [0.0, *(point["cost_usd"] for point in frontier)]
    speedups = [0.0, *(point["speedup_vs_triton"] for point in frontier)]
    axis.plot(costs, speedups, color=color, label=gpu_type, linewidth=2, marker="o")
    return costs[-1], speedups[-1], color


def _extend_frontier(axis, endpoint):
    cost, speedup, color = endpoint
    axis.plot(
        [cost, axis.get_xlim()[1]],
        [speedup, speedup],
        color=color,
        linewidth=2,
        scalex=False,
        scaley=False,
    )


def _format_axis(axis, title):
    axis.axhline(1.0, color="#2563eb", linestyle="--", linewidth=1)
    axis.set_title(title, fontsize=AXIS_TITLE_FONT_SIZE)
    axis.set_xlabel("Cumulative API cost (USD)", fontsize=AXIS_LABEL_FONT_SIZE)
    axis.set_ylabel("Speedup vs. Triton", fontsize=AXIS_LABEL_FONT_SIZE)
    axis.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    axis.grid(True, alpha=0.3)


def _save_figure(figure, output_directory, basename):
    saved_paths = []
    for output_format in PLOT_FORMATS:
        output_path = output_directory / f"{basename}.{output_format}"
        figure.savefig(output_path, format=output_format, dpi=300, bbox_inches="tight")
        saved_paths.append(output_path)
    return saved_paths


def write_frontier_grid(output_directory, accepted_kernels):
    kernels = defaultdict(lambda: defaultdict(list))
    for point in accepted_kernels:
        kernel_name = _display_kernel_name(str(point["kernel"]))
        series = (point["gpu_type"], point["precision"])
        kernels[kernel_name][series].append(point)
    kernel_names = sorted(kernels)
    if len(kernel_names) > GRID_SIZE:
        LOGGER.warning(
            "Plotting the first %d of %d kernels.", GRID_SIZE, len(kernel_names)
        )

    figure, axes = plt.subplots(
        GRID_ROWS,
        GRID_COLUMNS,
        figsize=(20, 13),
    )
    for index, axis in enumerate(axes.flat):
        if index < len(kernel_names) and index < GRID_SIZE:
            kernel_name = kernel_names[index]
            endpoints = []
            for color_index, ((gpu_type, precision), points) in enumerate(
                sorted(kernels[kernel_name].items())
            ):
                endpoints.append(
                    _plot_frontier(
                        axis,
                        pareto_frontier(points),
                        GPU_COLORS[color_index % len(GPU_COLORS)],
                        f"{gpu_type}: {precision}",
                    )
                )
            _format_axis(axis, kernel_name)
            for endpoint in endpoints:
                _extend_frontier(axis, endpoint)
            axis.legend()
        else:
            axis.set_visible(False)
    figure.subplots_adjust(hspace=0.45, wspace=0.3)
    saved_paths = _save_figure(figure, output_directory, GRID_PLOT_BASENAME)
    plt.close(figure)
    return saved_paths


def write_gemm_llm_frontier(output_directory, accepted_kernels):
    points_by_provider_and_model = defaultdict(lambda: defaultdict(list))
    for point in accepted_kernels:
        if (
            point["precision"] == "Floating Point 16"
            and _is_gemm_kernel(str(point["kernel"]))
            and _is_l40s_gemm_point(point)
        ):
            points_by_provider_and_model[point["provider"]][point["model"]].append(
                point
            )

    provider_groups = ("OpenAI LLMs", "Anthropic LLMs", "Other LLMs")
    figure, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
    for axis, provider_group in zip(axes, provider_groups, strict=True):
        models = points_by_provider_and_model[provider_group]
        endpoints = []
        for color_index, (model, points) in enumerate(sorted(models.items())):
            endpoints.append(
                _plot_frontier(
                    axis,
                    pareto_frontier(points),
                    GPU_COLORS[color_index % len(GPU_COLORS)],
                    model,
                )
            )
        _format_axis(axis, provider_group)
        axis.set_xlim(0.0, GEMM_COST_LIMIT_USD)
        for endpoint in endpoints:
            _extend_frontier(axis, endpoint)
        failed_models = GEMM_FAILED_MODELS_BY_PROVIDER.get(provider_group, ())
        if models or failed_models:
            legend_handles, legend_labels = axis.get_legend_handles_labels()
            for failed_model in failed_models:
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        color="#dc2626",
                        marker="x",
                        linestyle="None",
                        markersize=8,
                    )
                )
                legend_labels.append(f"{failed_model} (failed)")
            axis.legend(legend_handles, legend_labels, fontsize=12)
        else:
            axis.text(
                0.5,
                0.5,
                "No accepted GEMM results",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=13,
            )
    figure.subplots_adjust(wspace=0.2)
    saved_paths = _save_figure(
        figure,
        output_directory,
        GEMM_LLM_FRONTIER_PLOT_BASENAME,
    )
    plt.close(figure)
    return saved_paths


def write_token_expansion_plot(output_directory, token_expansion_data):
    kernel_names = sorted({data["kernel"] for data in token_expansion_data})
    gpu_types = sorted({data["gpu_type"] for data in token_expansion_data})
    data_by_kernel_and_gpu = {
        (data["kernel"], data["gpu_type"]): data for data in token_expansion_data
    }
    figure, axis = plt.subplots(figsize=(30, 12))
    kernel_indices = list(range(len(kernel_names)))
    bar_width = 0.8 / (2 * len(gpu_types))
    for gpu_index, gpu_type in enumerate(gpu_types):
        offset = -0.4 + (2 * gpu_index + 0.5) * bar_width
        triton_values = [
            data_by_kernel_and_gpu.get((kernel_name, gpu_type), {}).get(
                "triton_generated_expansion_factor", 0.0
            )
            for kernel_name in kernel_names
        ]
        llm_values = [
            data_by_kernel_and_gpu.get((kernel_name, gpu_type), {}).get(
                "llm_generated_expansion_factor", 0.0
            )
            for kernel_name in kernel_names
        ]
        triton_bars = axis.bar(
            [index + offset for index in kernel_indices],
            triton_values,
            bar_width,
            color=GPU_COLORS[gpu_index % len(GPU_COLORS)],
            label=f"{gpu_type}: Triton-generated PTX",
        )
        llm_bars = axis.bar(
            [index + offset + bar_width for index in kernel_indices],
            llm_values,
            bar_width,
            color=GPU_COLORS[gpu_index % len(GPU_COLORS)],
            hatch="//",
            label=f"{gpu_type}: LLM-generated PTX",
        )
        triton_labels = [
            f"{factor:.1f}x\n{data_by_kernel_and_gpu[(kernel_name, gpu_type)]['triton_generated_ptx_tokens']:,} tokens"
            if factor
            else ""
            for kernel_name, factor in zip(kernel_names, triton_values, strict=True)
        ]
        llm_labels = [
            f"{factor:.1f}x\n{data_by_kernel_and_gpu[(kernel_name, gpu_type)]['llm_generated_ptx_tokens']:,} tokens"
            if factor
            else ""
            for kernel_name, factor in zip(kernel_names, llm_values, strict=True)
        ]
        axis.bar_label(
            triton_bars,
            labels=triton_labels,
            padding=4,
            fontsize=11,
            rotation=90,
        )
        axis.bar_label(
            llm_bars,
            labels=llm_labels,
            padding=4,
            fontsize=11,
            rotation=90,
        )
    axis.set_xticks(kernel_indices, kernel_names)
    axis.set_ylabel("Token expansion factor vs. Triton source", fontsize=24)
    axis.tick_params(axis="x", labelrotation=30, labelsize=17)
    axis.tick_params(axis="y", labelsize=18)
    axis.grid(axis="y", alpha=0.3)
    axis.legend(fontsize=16, ncols=2)
    figure.subplots_adjust(bottom=0.25)
    saved_paths = _save_figure(
        figure,
        output_directory,
        TOKEN_EXPANSION_PLOT_BASENAME,
    )
    plt.close(figure)
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Plot speedup/cost Pareto frontiers from agent trace JSON."
    )
    parser.add_argument("trace_directory", type=Path, nargs="?", default="astra")
    parser.add_argument("--output-directory", type=Path, default="plots")
    arguments = parser.parse_args()
    trace_directory = arguments.trace_directory.resolve()
    if not trace_directory.is_dir():
        parser.error(f"Not a directory: {trace_directory}")

    accepted_kernels = load_accepted_kernels(trace_directory)
    if not accepted_kernels:
        raise RuntimeError("No accepted kernels with run_cost_usd_so_far were found.")

    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    grid_paths = write_frontier_grid(output_directory, accepted_kernels)
    gemm_llm_frontier_paths = write_gemm_llm_frontier(
        output_directory,
        accepted_kernels,
    )
    token_expansion_data = load_token_expansion_data(trace_directory)
    if not token_expansion_data:
        raise RuntimeError("No Triton-generated PTX files were found.")
    token_expansion_paths = write_token_expansion_plot(
        output_directory,
        token_expansion_data,
    )
    for output_path in [
        *grid_paths,
        *gemm_llm_frontier_paths,
        *token_expansion_paths,
    ]:
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
