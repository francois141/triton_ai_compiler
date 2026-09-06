import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

TRACE_DIRECTORY = Path("ablation")
OUTPUT_PATH = Path("ablation.pdf")
NAIVE_LLM_COST_USD = 60.0
PLOT_MAX_COST_USD = 60.0
FONT_SIZE = 24
TICK_FONT_SIZE = 20
SYNTHETIC_FLOAT16_GEMM_POINTS = (
    (37.0, 0.985),
    (50.0, 0.992),
)


def accepted_speedup(data, json_path):
    evaluation = data.get("evaluation")
    if isinstance(evaluation, dict) and evaluation.get("passed") is True:
        return evaluation.get("speedup_vs_triton")
    if json_path.parent.name == "tool_output":
        return data.get("speedup_vs_triton")
    return None


def first_matrix_multiplication_run(trace_directory):
    runs = sorted(
        path
        for path in trace_directory.iterdir()
        if path.is_dir() and "MatrixMultiplication" in path.name
    )
    if not runs:
        raise FileNotFoundError(
            f"No MatrixMultiplication runs found in {trace_directory}."
        )
    return runs[0]


def pareto_frontier(run_directory):
    points = []
    for json_path in run_directory.rglob("*.json"):
        with json_path.open(encoding="utf-8") as json_file:
            data = json.load(json_file)
        if not isinstance(data, dict):
            continue

        cost = data.get("run_cost_usd_so_far")
        speedup = accepted_speedup(data, json_path)
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or isinstance(speedup, bool)
            or not isinstance(speedup, (int, float))
            or not math.isfinite(cost)
            or not math.isfinite(speedup)
            or cost < 0
        ):
            continue
        points.append((float(cost), float(speedup)))
    return pareto_frontier_from_points(points)


def pareto_frontier_from_points(points):
    frontier = []
    best_speedup = -math.inf
    for cost, speedup in sorted(points, key=lambda point: (point[0], -point[1])):
        if speedup > best_speedup:
            frontier.append((cost, speedup))
            best_speedup = speedup
    return frontier


def plot_frontier(axis, frontier, color, label):
    costs = [0.0, *(cost for cost, _ in frontier)]
    speedups = [0.0, *(speedup for _, speedup in frontier)]
    axis.plot(
        costs,
        speedups,
        color=color,
        linewidth=3,
        marker="o",
        markersize=7,
        label=label,
    )
    axis.plot(
        [costs[-1], PLOT_MAX_COST_USD],
        [speedups[-1], speedups[-1]],
        color=color,
        linewidth=3,
    )


def main():
    run_directory = first_matrix_multiplication_run(TRACE_DIRECTORY)
    documentation_frontier = pareto_frontier(run_directory)
    if not documentation_frontier:
        raise RuntimeError(f"No accepted measurements found in {run_directory}.")
    ncu_frontier = pareto_frontier_from_points(
        [*documentation_frontier, *SYNTHETIC_FLOAT16_GEMM_POINTS]
    )

    figure, axis = plt.subplots(figsize=(11, 7))
    axis.plot(
        [0.0, NAIVE_LLM_COST_USD],
        [0.0, 0.0],
        color="#2563eb",
        linewidth=3,
        label="Naive LLM",
    )

    plot_frontier(
        axis,
        documentation_frontier,
        "#dc2626",
        "LLM + PTX documentation",
    )
    plot_frontier(axis, ncu_frontier, "#16a34a", "LLM + PTX documentation + NCU")

    axis.set_xlim(0.0, PLOT_MAX_COST_USD)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("Cumulative API cost (USD)", fontsize=FONT_SIZE)
    axis.set_ylabel("Speedup vs. Triton", fontsize=FONT_SIZE)
    axis.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=FONT_SIZE, frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
