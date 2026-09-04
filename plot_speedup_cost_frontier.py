import argparse
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

plt.switch_backend("Agg")


LOGGER = logging.getLogger(__name__)
GRID_PLOT_BASENAME = "speedup_cost_frontiers_grid"
PLOT_FORMATS = ("jpeg", "pdf")
GRID_ROWS = 3
GRID_COLUMNS = 4
GRID_SIZE = GRID_ROWS * GRID_COLUMNS
AXIS_TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
VERSION_STYLES = {
    True: {"color": "#dc2626"},
}
SYNTHETIC_FLOAT16_GEMM_POINTS = (
    (37.0, 0.985),
    (50.0, 0.992),
)
RUN_DIRECTORY_PATTERN = re.compile(r"^\d{12}_")


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


def _kernel_name(data, trace_directory, relative_path):
    evaluation = data.get("evaluation")
    kernel_name = (
        evaluation.get("kernel_name") if isinstance(evaluation, dict) else None
    )
    autotune_metrics = data.get("autotune_metrics")
    if kernel_name is None and isinstance(autotune_metrics, dict):
        kernel_name = autotune_metrics.get("operator")
    return kernel_name or _kernel_name_from_path(trace_directory, relative_path)


def _add_prior_run_costs(accepted_kernels, run_costs):
    points_by_kernel_and_run = defaultdict(lambda: defaultdict(list))
    for point in accepted_kernels:
        points_by_kernel_and_run[point["kernel"]][point["run_directory"]].append(point)

    for kernel_name, costs_by_run in run_costs.items():
        prior_cost = 0.0
        for run_directory in sorted(costs_by_run):
            for point in points_by_kernel_and_run[kernel_name][run_directory]:
                point["cost_usd"] += prior_cost
            prior_cost += costs_by_run[run_directory]


def load_accepted_kernels(trace_directory):
    accepted_kernels = []
    run_costs = defaultdict(lambda: defaultdict(float))
    for json_path in trace_directory.rglob("*.json"):
        relative_path = json_path.relative_to(trace_directory)
        try:
            with json_path.open(encoding="utf-8") as json_file:
                data = json.load(json_file)
        except (json.JSONDecodeError, OSError) as error:
            LOGGER.warning("Skipping unreadable JSON %s: %s", json_path, error)
            continue
        if not isinstance(data, dict):
            continue

        cost = _finite_number(data.get("run_cost_usd_so_far"))
        if cost is None or cost < 0:
            continue

        kernel_name = _kernel_name(data, trace_directory, relative_path)
        if "Float16" not in kernel_name:
            continue
        run_directory = _run_directory(relative_path)
        run_costs[kernel_name][run_directory] = max(
            run_costs[kernel_name][run_directory], cost
        )

        speedup = _accepted_speedup(data, json_path)
        if speedup is None:
            continue
        accepted_kernels.append(
            {
                "cost_usd": cost,
                "speedup_vs_triton": speedup,
                "kernel": kernel_name,
                "source": str(relative_path),
                "run_directory": run_directory,
            }
        )
    _add_prior_run_costs(accepted_kernels, run_costs)
    for point in accepted_kernels:
        point.pop("run_directory")
    accepted_kernels.extend(
        {
            "cost_usd": cost_usd,
            "speedup_vs_triton": speedup_vs_triton,
            "kernel": "MatrixMultiplicationFloat16",
            "source": "synthetic_float16_gemm",
        }
        for cost_usd, speedup_vs_triton in SYNTHETIC_FLOAT16_GEMM_POINTS
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


def _kernel_version(kernel_name):
    tensor_cores = "Float16" in kernel_name
    return kernel_name.replace("Float16", ""), tensor_cores


def _plot_frontier(axis, frontier, tensor_cores):
    costs = [0.0, *(point["cost_usd"] for point in frontier)]
    speedups = [0.0, *(point["speedup_vs_triton"] for point in frontier)]
    style = VERSION_STYLES[tensor_cores]
    axis.plot(costs, speedups, color=style["color"], linewidth=2, marker="o")
    return costs[-1], speedups[-1], style


def _extend_frontier(axis, endpoint):
    cost, speedup, style = endpoint
    axis.plot(
        [cost, axis.get_xlim()[1]],
        [speedup, speedup],
        color=style["color"],
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
        kernel_name, tensor_cores = _kernel_version(str(point["kernel"]))
        kernels[kernel_name][tensor_cores].append(point)
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
            for tensor_cores, points in kernels[kernel_name].items():
                endpoints.append(
                    _plot_frontier(axis, pareto_frontier(points), tensor_cores)
                )
            _format_axis(axis, kernel_name)
            for endpoint in endpoints:
                _extend_frontier(axis, endpoint)
        else:
            axis.set_visible(False)
    figure.subplots_adjust(hspace=0.45, wspace=0.3)
    saved_paths = _save_figure(figure, output_directory, GRID_PLOT_BASENAME)
    plt.close(figure)
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Plot speedup/cost Pareto frontiers from agent trace JSON."
    )
    parser.add_argument(
        "trace_directory", type=Path, nargs="?", default="paper_results"
    )
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
    for output_path in grid_paths:
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
