import argparse
import ast
import json
import logging
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import tiktoken
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from utils.clean_ptx import clean_ptx

plt.switch_backend("Agg")


LOGGER = logging.getLogger(__name__)
GRID_PLOT_BASENAME = "speedup_cost_frontiers_grid"
TOKEN_EXPANSION_PLOT_BASENAME = "ptx_token_expansion_factors"
GEMM_LLM_FRONTIER_OPENAI_PLOT_BASENAME = "gemm_llm_cost_frontier_openai"
SPEEDUP_ORIGINAL_PLOT_BASENAME = "speedup_original"
CONFERENCE_KERNEL_SPEEDUP_PLOT_BASENAME = "conference_kernel_speedups_b200"
FIVE_TRIAL_TABLE_BASENAME = "five_trial_speedups_b200"
FIVE_TRIAL_TABLE_FILENAME = f"{FIVE_TRIAL_TABLE_BASENAME}.tex"
REPRESENTATIVE_RESULTS_TABLE_FILENAME = "representative_kernel_results.tex"
GEMM_GPU_TYPE = "NVIDIA L40S"
GEMM_COST_LIMIT_USD = 15.0
# Restrict this set of figures to GPT-6 experiment traces.
MODEL_PREFIXES = ("gpt-6",)
GEMM_FAILED_OPENAI_MODELS = ("gpt-5.6-luna",)
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
AXIS_TITLE_FONT_SIZE = 22
AXIS_LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 17
GPU_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# Fix the colour of each (GPU, precision) series so a series looks the same in
# every grid panel and matches the single shared legend.  Panels do not all
# contain the same series, so per-panel colour cycling would break the legend.
GRID_SERIES_COLORS = {
    ("NVIDIA B200", "Floating Point 16"): GPU_COLORS[0],
    ("NVIDIA B200", "Floating Point 8"): GPU_COLORS[1],
    ("NVIDIA H100", "Floating Point 16"): GPU_COLORS[3],
    ("NVIDIA H100", "Floating Point 8"): GPU_COLORS[4],
    ("NVIDIA L40S", "Floating Point 16"): GPU_COLORS[2],
    ("NVIDIA L40S", "Floating Point 8"): GPU_COLORS[6],
}
GRID_FALLBACK_SERIES_COLOR = GPU_COLORS[7 % len(GPU_COLORS)]
# Only show a new frontier point when it is visibly better than the previous
# displayed point.  This prevents tiny benchmark fluctuations from making
# otherwise comparable panels look more densely sampled.
MIN_PLOTTED_SPEEDUP_GAIN = 0.01
FRONTIER_LINE_WIDTH = 4.0
FRONTIER_MARKER_SIZE = 11.0
# Edit the strings on the right to change the 12 grid-panel titles.  Keys are
# the trace-derived kernel names after their Float16/Float8 suffix is removed.
GRID_TITLE_TRANSLATIONS = {
    "Convolution2DKernel": "Convolution 2D",
    "FusedGEMMAddSiLUKernel": "Fused GEMM + SiLU",
    "GELUKernel": "GELU",
    "SigmoidKernel": "Sigmoid",
    "MatrixMultiplication": "Matrix Multiplication",
    "MatrixVectorMultiplicationKernel": "Matrix-Vector Multiplication",
    "RMSNormKernel": "RMSNorm",
    "ReLUKernel": "ReLU",
    "ReductionSumKernel": "Reduction Sum",
    "RoPEKernel": "RoPE",
    "SiLUKernel": "SiLU",
    "SoftmaxKernel": "Softmax",
    "SwiGLUKernel": "SwiGLU",
}
# Narrative order for the 3x4 grid: matrix-oriented kernels, activations, then
# normalization/reduction/positioning kernels. Edit this tuple to reorder panels.
GRID_KERNEL_ORDER = (
    "Convolution2DKernel",
    "FusedGEMMAddSiLUKernel",
    "MatrixMultiplication",
    "MatrixVectorMultiplicationKernel",
    "SigmoidKernel",
    "ReLUKernel",
    "SiLUKernel",
    "SwiGLUKernel",
    "RMSNormKernel",
    "ReductionSumKernel",
    "RoPEKernel",
    "SoftmaxKernel",
)
# Panels within a row share a Y scale; each row is scaled to its workload family.
GRID_Y_LIMITS_BY_ROW = ((0.5, 2.5), (0.5, 1.15), (0.5, 1.35))
# The paper figure has a deliberate narrative order rather than alphabetical
# ordering. Measurements remain fully data-driven: a kernel is omitted when
# no accepted B200 result is present in the supplied traces.
CONFERENCE_KERNEL_PLOT_ORDER = (
    "BitDeltaNeurIPS2024Matmul",
    "BitDeltaNeurIPS2024BatchedMatmul",
    "Dion2TritonPostOrthogonalize",
    "FlashAttentionNeurIPS2022Forward",
    "FlashSinkhornFusedSchurMatvec",
    "ForgettingAttentionICLR2025Forward",
    "LionNeurIPS2023Optimizer",
    "Mamba2ChunkScanForward",
    "Mamba2ChunkStateForward",
    "SageAttentionICLR2025",
)
# Edit this table to set the displayed title and venue beneath each bar.
# The keys are trace kernel names; use \n in either string to add a line break.
CONFERENCE_KERNEL_FIGURE_TITLES = {
    "BitDeltaNeurIPS2024Matmul": ("BitDelta", "NeurIPS 2024"),
    "BitDeltaNeurIPS2024BatchedMatmul": ("BitDelta\nbatched", "NeurIPS 2024"),
    "Dion2TritonPostOrthogonalize": ("Dion2", "Microsoft Research\nForum 2026"),
    "FlashAttentionNeurIPS2022Forward": ("FlashAttention", "NeurIPS 2022"),
    "FlashSinkhornFusedSchurMatvec": ("FlashSinkhorn", "ICML 2026"),
    "ForgettingAttentionICLR2025Forward": ("ForgettingAttention", "ICLR 2025"),
    "LionNeurIPS2023Optimizer": ("Lion optimizer", "NeurIPS 2023"),
    "Mamba2ChunkScanForward": ("Mamba-2 chunk\nscan forward", "ICML 2024"),
    "Mamba2ChunkStateForward": ("Mamba-2 chunk\nstate forward", "ICML 2024"),
    "SageAttentionICLR2025": ("SageAttention", "ICLR 2025"),
}
CONFERENCE_KERNEL_DISPLAY_NAMES = {
    kernel_name: f"{operator_label.replace(chr(10), ' ')} ({venue_label.replace(chr(10), ' ')})"
    for kernel_name, (operator_label, venue_label) in CONFERENCE_KERNEL_FIGURE_TITLES.items()
}
FLOAT_PRECISION_LABELS = {
    "Float16": "Floating Point 16",
    "Float8": "Floating Point 8",
}
REPRESENTATIVE_KERNELS = (
    ("Fused GEMM + SiLU", "FusedGEMMAddSiLUFloat16Kernel"),
    ("Softmax", "SoftmaxFloat16Kernel"),
    ("Matrix multiplication", "MatrixMultiplicationFloat16"),
)
REPRESENTATIVE_GPU_TYPES = ("NVIDIA H100", "NVIDIA B200", "NVIDIA L40S")
FIVE_TRIAL_COUNT = 5
FIVE_TRIAL_KERNELS = (
    ("Convolution (FP8)", "Convolution2DFloat8Kernel", 5.0),
    ("Matrix multiplication (FP16)", "MatrixMultiplicationFloat16", 5.0),
    ("SwiGLU (FP16)", "SwiGLUFloat16Kernel", 1.0),
    (
        "FlashAttention forward (FP16)",
        "FlashAttentionNeurIPS2022Forward",
        5.0,
    ),
)
RUN_DIRECTORY_PATTERN = re.compile(r"^\d{12}_")
# Duplicated trace directories (for example "..._max copy") hold the same
# measurements as their original and must not be counted or plotted twice.
DUPLICATE_RUN_SUFFIX = " copy"
REASONING_EFFORT_PATTERN = re.compile(
    r"_(?:low|medium|high|max|xhigh|ultra|none)$", re.IGNORECASE
)
KERNEL_SOURCE_DIRECTORY = (
    Path(__file__).resolve().parent / "triton_ptx" / "triton_ptx" / "kernels"
)
ASTRA_TRACE_DIRECTORY = Path(__file__).resolve().parent / "astra"
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


def _is_duplicate_run(run_directory):
    return run_directory.name.endswith(DUPLICATE_RUN_SUFFIX)


def _is_included_run(run_directory, model_prefixes):
    if _is_duplicate_run(run_directory):
        return False
    return model_prefixes is None or _model_from_run_directory(
        run_directory
    ).startswith(model_prefixes)


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


def load_accepted_kernels(
    trace_directory,
    model_prefixes=MODEL_PREFIXES,
    *,
    include_non_floating_point=False,
):
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
        if not _is_included_run(run_directory, model_prefixes):
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
        # Most existing figures compare Float16/Float8 workloads. Conference
        # kernels use paper-specific class names instead, so allow the B200
        # conference chart to request those results explicitly.
        if precision is None and not include_non_floating_point:
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
                # Cost within this run only.  "cost_usd" additionally
                # accumulates the cost of earlier runs of the same kernel.
                "run_cost_usd": cost,
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


def plotted_frontier(points, min_relative_gain=MIN_PLOTTED_SPEEDUP_GAIN):
    """Return Pareto points separated by at least ``min_relative_gain``.

    The threshold is relative: a point at 1.010x is retained after 1.000x,
    while a point below that is omitted until the cumulative improvement is
    large enough to be meaningful in the plot.
    """
    displayed = []
    for point in pareto_frontier(points):
        if not displayed or point["speedup_vs_triton"] >= (
            displayed[-1]["speedup_vs_triton"] * (1 + min_relative_gain)
        ):
            displayed.append(point)
    return displayed


def _display_kernel_name(kernel_name):
    for precision in FLOAT_PRECISION_LABELS:
        kernel_name = kernel_name.replace(precision, "")
    return kernel_name


def _display_gpu_type(gpu_type):
    """Hide H100 memory/interconnect variants in plot labels."""
    if gpu_type.startswith("NVIDIA H100"):
        return "NVIDIA H100"
    return gpu_type


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
        # The fused GEMM + SiLU benchmark is retained as an experiment trace
        # rather than a checked-in Triton kernel class.  Its prompt records the
        # exact Triton source used for compilation, which is sufficient for the
        # token-expansion analysis.
        if kernel_name.startswith("FusedGEMMAddSiLU"):
            for prompt_path in sorted(
                ASTRA_TRACE_DIRECTORY.glob(
                    f"*{kernel_name}*/iteration_000_try_00_initial_candidate_prompt.txt"
                )
            ):
                prompt = prompt_path.read_text(encoding="utf-8")
                match = re.search(
                    r"## Triton Kernel\s*```python\s*(.*?)\s*```",
                    prompt,
                    flags=re.DOTALL,
                )
                if match is not None:
                    return match.group(1)
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
        if not _is_included_run(run_directory, MODEL_PREFIXES):
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
                # Treat H100 memory/interconnect variants as a single device
                # series. The token-expansion runs are complementary across
                # the variants, so this keeps every kernel while avoiding
                # duplicate H100 legend entries.
                "gpu_type": _display_gpu_type(
                    gpu_types_by_run.get(run_directory, "Unknown GPU")
                ),
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


def _plot_frontier(axis, frontier, color, gpu_type, *, include_origin=True):
    costs = [point["cost_usd"] for point in frontier]
    speedups = [point["speedup_vs_triton"] for point in frontier]
    if include_origin:
        costs.insert(0, 0.0)
        speedups.insert(0, 0.0)
    axis.plot(
        costs,
        speedups,
        color=color,
        label=gpu_type,
        linewidth=FRONTIER_LINE_WIDTH,
        marker="o",
        markersize=FRONTIER_MARKER_SIZE,
    )
    return costs[-1], speedups[-1], color


def _extend_frontier(axis, endpoint):
    cost, speedup, color = endpoint
    axis.plot(
        [cost, axis.get_xlim()[1]],
        [speedup, speedup],
        color=color,
        linewidth=FRONTIER_LINE_WIDTH,
        scalex=False,
        scaley=False,
    )


def _format_axis(axis, title, show_axis_labels=True):
    axis.axhline(1.0, color="#2563eb", linestyle="--", linewidth=1)
    # Keep the enlarged panel headings inside their own panels.
    if len(title) > 18 and title.endswith("Kernel"):
        title = f"{title[:-6]}\nKernel"
    axis.set_title(title, fontsize=AXIS_TITLE_FONT_SIZE, fontweight="bold")
    if show_axis_labels:
        axis.set_xlabel("Cumulative API cost (USD)", fontsize=AXIS_LABEL_FONT_SIZE)
        axis.set_ylabel("Speedup vs. Triton", fontsize=AXIS_LABEL_FONT_SIZE)
    axis.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    axis.grid(True, alpha=0.3)


def _format_cost_ticks_as_usd(axis):
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: rf"\${value:g}")
    )


def _save_figure(figure, output_directory, basename):
    saved_paths = []
    for output_format in PLOT_FORMATS:
        output_path = output_directory / f"{basename}.{output_format}"
        figure.savefig(output_path, format=output_format, dpi=300, bbox_inches="tight")
        saved_paths.append(output_path)
    return saved_paths


def _best_speedup_for_run(run_directory):
    best_speedup = None
    gpu_type = None
    pending_gpu_type = None
    for json_path in run_directory.rglob("*.json"):
        try:
            with json_path.open(encoding="utf-8") as json_file:
                data = json.load(json_file)
        except (json.JSONDecodeError, OSError) as error:
            LOGGER.warning("Skipping unreadable JSON %s: %s", json_path, error)
            continue
        if json_path.name in PENDING_SPEEDUP_EVENTS_FILENAMES:
            pending_gpu_type = _pending_events_gpu_type(data) or pending_gpu_type
            continue
        if not isinstance(data, dict):
            continue
        gpu_type = _gpu_type(data) or gpu_type
        speedup = _accepted_speedup(data, json_path)
        if speedup is not None:
            best_speedup = (
                speedup if best_speedup is None else max(best_speedup, speedup)
            )
    if not _is_b200_gpu(gpu_type or pending_gpu_type or ""):
        return None
    return best_speedup


def load_five_trial_results(trace_directory):
    results = []
    for display_name, kernel_name, budget in FIVE_TRIAL_KERNELS:
        trial_results = []
        for run_directory in sorted(trace_directory.glob(f"*_{kernel_name}_*")):
            if not run_directory.is_dir() or run_directory.name.endswith(" copy"):
                continue
            speedup = _best_speedup_for_run(run_directory)
            if speedup is not None:
                trial_results.append((run_directory.name, speedup))
        if len(trial_results) < FIVE_TRIAL_COUNT:
            raise RuntimeError(
                f"Expected {FIVE_TRIAL_COUNT} B200 trials for {kernel_name}; "
                f"found {len(trial_results)}."
            )
        if len(trial_results) > FIVE_TRIAL_COUNT:
            LOGGER.warning(
                "Using the latest %d of %d B200 trials for %s.",
                FIVE_TRIAL_COUNT,
                len(trial_results),
                kernel_name,
            )
        speedups = [speedup for _, speedup in trial_results[-FIVE_TRIAL_COUNT:]]
        results.append(
            {
                "display_name": display_name,
                "budget": budget,
                "speedups": speedups,
                "mean": statistics.fmean(speedups),
                "stdev": statistics.stdev(speedups),
                "best": max(speedups),
            }
        )
    return results


def write_five_trial_results_table(output_directory, results):
    headers = [
        "Kernel",
        "Budget",
        *(f"Trial {index}" for index in range(1, FIVE_TRIAL_COUNT + 1)),
        "Mean ± SD",
        "Best",
    ]
    rows = [
        [
            result["display_name"],
            f"${result['budget']:g}",
            *(f"{speedup:.3f}x" for speedup in result["speedups"]),
            f"{result['mean']:.3f} ± {result['stdev']:.3f}",
            f"{result['best']:.3f}x",
        ]
        for result in results
    ]
    figure, axis = plt.subplots(figsize=(16, 3.25))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.22, 0.08, *([0.075] * FIVE_TRIAL_COUNT), 0.16, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.8)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        if row == 0:
            cell.set_facecolor("#E8F0F8")
            cell.set_text_props(weight="bold", color="#1F2937")
        elif row % 2 == 0:
            cell.set_facecolor("#F8FAFC")
        if column == 0:
            cell.set_text_props(ha="left", weight="bold")
    figure.suptitle(
        "Five independent neural-lowering trials on NVIDIA B200",
        fontsize=15,
        fontweight="bold",
        y=0.96,
    )
    axis.set_title(
        "Speedup versus Triton; trials are ordered chronologically",
        fontsize=10.5,
        color="#4B5563",
        pad=10,
    )
    figure.subplots_adjust(left=0.015, right=0.985, top=0.79, bottom=0.05)
    saved_paths = _save_figure(figure, output_directory, FIVE_TRIAL_TABLE_BASENAME)
    plt.close(figure)

    latex_rows = []
    for result in results:
        trials = " & ".join(
            f"${speedup:.3f}\\times$" for speedup in result["speedups"]
        )
        latex_rows.append(
            f"{result['display_name']} & \\${result['budget']:g} & {trials} & "
            f"${result['mean']:.3f} \\pm {result['stdev']:.3f}$ & "
            f"${result['best']:.3f}\\times$ \\\\"
        )
    latex_table = "\n".join(
        (
            r"\begin{table*}[h]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}lcrrrrrrr@{}}",
            r"\toprule",
            r"\textbf{Kernel} & \textbf{Budget} & \textbf{Trial 1}",
            r"& \textbf{Trial 2} & \textbf{Trial 3} & \textbf{Trial 4}",
            r"& \textbf{Trial 5} & \textbf{Mean $\pm$ SD} & \textbf{Best} \\",
            r"\midrule",
            *latex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Repeatability on NVIDIA B200 across five independent",
            r"\neuralcompiler{} runs. Values are speedups over the same autotuned",
            r"Triton baseline. Budget is the configured per-run API-cost gate, checked",
            r"before each new model request; an in-flight request can finish above it.}",
            r"\label{tab:five-trial-speedups}",
            r"\end{table*}",
            "",
        )
    )
    latex_path = output_directory / FIVE_TRIAL_TABLE_FILENAME
    latex_path.write_text(latex_table, encoding="utf-8")
    return [latex_path, *saved_paths]


def _ptx_path_for_accepted_point(trace_directory, point):
    json_path = trace_directory / point["source"]
    candidate_paths = (
        json_path.with_suffix(".ptx"),
        json_path.parent / "final_candidate.ptx",
        *json_path.parent.glob(f"{json_path.stem}_speedup_vs_triton_*.ptx"),
    )
    for ptx_path in candidate_paths:
        if ptx_path.is_file():
            return ptx_path
    raise RuntimeError(f"No PTX artifact found for accepted result {json_path}.")


def _representative_kernel_results(trace_directory, accepted_kernels):
    results = []
    for display_name, kernel_name in REPRESENTATIVE_KERNELS:
        triton_source = _triton_kernel_source(kernel_name)
        if triton_source is None:
            raise RuntimeError(f"No Triton source found for {kernel_name}.")
        triton_tokens = _token_count(triton_source)
        if triton_tokens == 0:
            raise RuntimeError(f"Triton source for {kernel_name} has no tokens.")
        results_by_gpu = []
        for gpu_type in REPRESENTATIVE_GPU_TYPES:
            candidates = [
                point
                for point in accepted_kernels
                if point["kernel"] == kernel_name
                and point["precision"] == "Floating Point 16"
                and _display_gpu_type(point["gpu_type"]) == gpu_type
            ]
            if not candidates:
                raise RuntimeError(
                    f"No accepted FP16 result found for {kernel_name} on {gpu_type}."
                )
            best_point = max(candidates, key=lambda point: point["speedup_vs_triton"])
            ptx_path = _ptx_path_for_accepted_point(trace_directory, best_point)
            expansion = (
                _token_count(ptx_path.read_text(encoding="utf-8")) / triton_tokens
            )
            results_by_gpu.append((expansion, best_point["speedup_vs_triton"]))
        results.append((display_name, results_by_gpu))
    return results


def write_representative_results_table(
    output_directory, trace_directory, accepted_kernels
):
    """Write trace-derived representative results as a LaTex table."""
    rows = []
    for kernel_name, results_by_gpu in _representative_kernel_results(
        trace_directory, accepted_kernels
    ):
        values = " & ".join(
            f"${value:.{2 if index % 2 else 1}f}\\times$"
            for result in results_by_gpu
            for index, value in enumerate(result)
        )
        rows.append(f"{kernel_name:<24} & {values} \\\\")
    table = "\n".join(
        (
            r"\begin{table*}[h]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{8pt}",
            r"\begin{tabular}{@{}lrrrrrr@{}}",
            r"\toprule",
            r"& \multicolumn{2}{c}{\textbf{NVIDIA H100}}",
            r"& \multicolumn{2}{c}{\textbf{NVIDIA B200}}",
            r"& \multicolumn{2}{c}{\textbf{NVIDIA L40S}} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
            r"\textbf{Kernel}",
            r"& \textbf{Expansion} & \textbf{Speedup}",
            r"& \textbf{Expansion} & \textbf{Speedup}",
            r"& \textbf{Expansion} & \textbf{Speedup} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Representative neural-lowering results on three GPU architectures.",
            r"Expansion is the ratio of LLM-generated PTX tokens to Triton-source tokens;",
            r"speedup is relative to the fastest correct Triton baseline. A dash denotes an",
            r"unavailable measurement.}",
            r"\label{tab:representative-kernel-results}",
            r"\end{table*}",
            "",
        )
    )
    output_path = output_directory / REPRESENTATIVE_RESULTS_TABLE_FILENAME
    output_path.write_text(table, encoding="utf-8")
    return output_path


def _grid_series_color(series):
    return GRID_SERIES_COLORS.get(series, GRID_FALLBACK_SERIES_COLOR)


def _grid_series_by_kernel(accepted_kernels):
    """Group frontier points by kernel and displayed (GPU, precision) series.

    One kernel and configuration is usually compiled several times: repeated
    runs, reasoning-effort variants, and devices that share a displayed name
    (the H100 PCIe and HBM3 parts both appear as "NVIDIA H100").  Each run is
    an independent compilation, so show the single best run instead of merging
    them: merging draws several identically labelled lines and chains the cost
    of unrelated runs onto one frontier, overstating what reaching a speedup
    costs.  Costs are therefore reported within the plotted run.
    """
    points_by_run = defaultdict(lambda: defaultdict(list))
    for point in accepted_kernels:
        kernel_name = _display_kernel_name(str(point["kernel"]))
        series = (_display_gpu_type(point["gpu_type"]), point["precision"])
        points_by_run[kernel_name][series, point["run_directory"]].append(point)

    kernels = defaultdict(dict)
    for kernel_name, points_by_series_and_run in points_by_run.items():
        best_by_series = {}
        for (series, _), points in sorted(points_by_series_and_run.items()):
            best_speedup = max(point["speedup_vs_triton"] for point in points)
            if series not in best_by_series or best_speedup > best_by_series[series][0]:
                best_by_series[series] = (
                    best_speedup,
                    [{**point, "cost_usd": point["run_cost_usd"]} for point in points],
                )
        for series, (_, points) in best_by_series.items():
            kernels[kernel_name][series] = points
    return kernels


def write_frontier_grid(
    output_directory,
    accepted_kernels,
    *,
    basename=GRID_PLOT_BASENAME,
    show_axis_labels=True,
    single_legend=True,
):
    kernels = _grid_series_by_kernel(accepted_kernels)
    kernel_names = [name for name in GRID_KERNEL_ORDER if name in kernels]
    kernel_names.extend(sorted(set(kernels) - set(kernel_names)))
    if len(kernel_names) > GRID_SIZE:
        LOGGER.warning(
            "Plotting the first %d of %d kernels.", GRID_SIZE, len(kernel_names)
        )

    plotted_series = {
        kernel_name: {
            series: plotted_frontier(points)
            for series, points in series_by_gpu_and_precision.items()
        }
        for kernel_name, series_by_gpu_and_precision in kernels.items()
    }
    displayed_points = [
        point
        for series_by_gpu_and_precision in plotted_series.values()
        for points in series_by_gpu_and_precision.values()
        for point in points
    ]
    if not displayed_points:
        raise RuntimeError("No Pareto frontier points were available to plot.")
    figure, axes = plt.subplots(
        GRID_ROWS,
        GRID_COLUMNS,
        figsize=(24, 16),
    )
    figure_legend_handles = []
    figure_legend_labels = []
    for index, axis in enumerate(axes.flat):
        if index < len(kernel_names) and index < GRID_SIZE:
            kernel_name = kernel_names[index]
            endpoints = []
            for series, points in sorted(plotted_series[kernel_name].items()):
                gpu_type, precision = series
                endpoints.append(
                    _plot_frontier(
                        axis,
                        points,
                        _grid_series_color(series),
                        f"{gpu_type}: {precision}",
                        include_origin=False,
                    )
                )
            title = GRID_TITLE_TRANSLATIONS.get(kernel_name, kernel_name)
            _format_axis(axis, title, show_axis_labels=show_axis_labels)
            axis.set_ylim(*GRID_Y_LIMITS_BY_ROW[index // GRID_COLUMNS])
            _format_cost_ticks_as_usd(axis)
            if single_legend:
                legend_handles, legend_labels = axis.get_legend_handles_labels()
                figure_legend_handles.extend(legend_handles)
                figure_legend_labels.extend(legend_labels)
            else:
                axis.legend()
        else:
            axis.set_visible(False)
    if single_legend and figure_legend_handles:
        unique_legend = dict(
            sorted(
                zip(figure_legend_labels, figure_legend_handles),
                key=lambda entry: entry[0],
            )
        )
        figure.legend(
            unique_legend.values(),
            unique_legend.keys(),
            loc="lower center",
            ncols=3,
            fontsize=16,
        )
        figure.supxlabel("Cumulative API cost (USD)", fontsize=AXIS_LABEL_FONT_SIZE,
                          fontweight="bold", y=0.075)
        figure.supylabel("Speedup vs. Triton", fontsize=AXIS_LABEL_FONT_SIZE,
                          fontweight="bold", x=0.015)
        figure.subplots_adjust(left=0.075, bottom=0.15, hspace=0.30, wspace=0.16)
    else:
        figure.subplots_adjust(hspace=0.32, wspace=0.18)
    saved_paths = _save_figure(figure, output_directory, basename)
    plt.close(figure)
    return saved_paths


def write_gemm_llm_frontier_openai(output_directory, accepted_kernels):
    """Plot the FP16 GEMM cost frontier for OpenAI models only."""
    points_by_model = defaultdict(list)
    for point in accepted_kernels:
        if (
            point["precision"] == "Floating Point 16"
            and _is_gemm_kernel(str(point["kernel"]))
            and _is_l40s_gemm_point(point)
            and point["provider"] == "OpenAI LLMs"
        ):
            points_by_model[point["model"]].append(point)

    figure, axis = plt.subplots(figsize=(9, 6))
    endpoints = []
    for color_index, (model, points) in enumerate(sorted(points_by_model.items())):
        endpoints.append(
            _plot_frontier(
                axis,
                pareto_frontier(points),
                GPU_COLORS[color_index % len(GPU_COLORS)],
                model,
            )
        )
    _format_axis(axis, "OpenAI LLMs")
    axis.set_xlim(0.0, GEMM_COST_LIMIT_USD)
    for endpoint in endpoints:
        _extend_frontier(axis, endpoint)
    legend_handles, legend_labels = axis.get_legend_handles_labels()
    for failed_model in GEMM_FAILED_OPENAI_MODELS:
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
    figure.tight_layout()
    saved_paths = _save_figure(
        figure,
        output_directory,
        GEMM_LLM_FRONTIER_OPENAI_PLOT_BASENAME,
    )
    plt.close(figure)
    return saved_paths


def _is_b200_gpu(gpu_type):
    return "b200" in gpu_type.casefold()


def plot_llm_figure(
    output_directory, accepted_kernels
):
    """Write the paper's B200 conference-kernel speedup figure from traces."""
    best_speedups = {}
    for point in accepted_kernels:
        kernel_name = str(point["kernel"])
        if not (
            _is_b200_gpu(str(point["gpu_type"]))
            and kernel_name in CONFERENCE_KERNEL_DISPLAY_NAMES
        ):
            continue
        best_speedups[kernel_name] = max(
            best_speedups.get(kernel_name, -math.inf),
            point["speedup_vs_triton"],
        )

    if not best_speedups:
        LOGGER.warning("No accepted conference-kernel results were found on B200.")
        return []

    kernel_results = [
        (*CONFERENCE_KERNEL_FIGURE_TITLES[kernel_name], best_speedups[kernel_name])
        for kernel_name in CONFERENCE_KERNEL_PLOT_ORDER
        if kernel_name in best_speedups
    ]
    labels, venues, speedups = zip(*kernel_results)
    indices = range(len(kernel_results))
    figure, axis = plt.subplots(figsize=(13.2, 5.6))
    bars = axis.bar(indices, speedups, width=0.78, color="#2C7FB8", edgecolor="white", linewidth=0.8)
    axis.axhline(1.0, color="#4E79A7", linestyle="--", linewidth=1.15, zorder=0)
    axis.text(len(kernel_results) - 0.25, 1.045, "parity", color="#4E79A7", fontsize=8, ha="right", va="bottom")
    axis.set_xlim(-0.65, len(kernel_results) - 0.35)
    axis.set_ylim(0, max(1.8, max(speedups) + 0.28))
    axis.set_ylabel("Best speedup vs. Triton", fontsize=13, fontweight="bold")
    axis.set_title("Best LLM speedup for conference kernels on NVIDIA B200", pad=8, fontsize=16, fontweight="bold")
    axis.set_xticks([])
    axis.tick_params(axis="both", labelsize=11)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)

    for index, (bar, speedup, label, venue) in enumerate(zip(bars, speedups, labels, venues)):
        axis.text(bar.get_x() + bar.get_width() / 2, speedup + 0.055, f"{speedup:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")
        label_y = -0.115 if index % 2 == 0 else -0.255
        axis.text(index, label_y, label, transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=9.1, fontweight="bold", linespacing=1.05, clip_on=False)
        axis.text(index, label_y - 0.052 * (label.count("\n") + 1), venue, transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=8.2, fontweight="medium", color="#5B6573", linespacing=1.05, clip_on=False)

    figure.subplots_adjust(left=0.09, right=0.995, top=0.88, bottom=0.34)
    saved_paths = _save_figure(
        figure,
        output_directory,
        CONFERENCE_KERNEL_SPEEDUP_PLOT_BASENAME,
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
    all_accepted_kernels = load_accepted_kernels(trace_directory, model_prefixes=None)

    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    representative_results_table_path = write_representative_results_table(
        output_directory, trace_directory, accepted_kernels
    )
    grid_paths = write_frontier_grid(
        output_directory, accepted_kernels, show_axis_labels=False
    )
    speedup_original_paths = write_frontier_grid(
        output_directory,
        accepted_kernels,
        basename=SPEEDUP_ORIGINAL_PLOT_BASENAME,
        show_axis_labels=True,
        single_legend=False,
    )
    gemm_llm_frontier_paths = write_gemm_llm_frontier_openai(
        output_directory,
        all_accepted_kernels,
    )
    conference_kernel_speedup_paths = plot_llm_figure(
        output_directory,
        load_accepted_kernels(
            trace_directory,
            include_non_floating_point=True,
        ),
    )
    five_trial_table_paths = write_five_trial_results_table(
        output_directory,
        load_five_trial_results(trace_directory),
    )
    token_expansion_data = load_token_expansion_data(trace_directory)
    if not token_expansion_data:
        raise RuntimeError("No Triton-generated PTX files were found.")
    token_expansion_paths = write_token_expansion_plot(
        output_directory,
        token_expansion_data,
    )
    for output_path in [
        representative_results_table_path,
        *grid_paths,
        *gemm_llm_frontier_paths,
        *conference_kernel_speedup_paths,
        *five_trial_table_paths,
        *speedup_original_paths,
        *token_expansion_paths,
    ]:
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
