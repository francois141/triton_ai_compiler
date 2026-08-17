import argparse
import html
import json
import logging
import math
from pathlib import Path

LOGGER = logging.getLogger(__name__)
PLOT_FILENAME = "speedup_cost_frontier.svg"


def _finite_number(value):
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
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


def load_accepted_kernels(trace_directory):
    accepted_kernels = []
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

        speedup = _accepted_speedup(data, json_path)
        cost = _finite_number(data.get("run_cost_usd_so_far"))
        if speedup is None or cost is None or cost < 0:
            continue

        evaluation = data.get("evaluation")
        kernel_name = None
        if isinstance(evaluation, dict):
            kernel_name = evaluation.get("kernel_name")
        autotune_metrics = data.get("autotune_metrics")
        if kernel_name is None and isinstance(autotune_metrics, dict):
            kernel_name = autotune_metrics.get("operator")
        accepted_kernels.append(
            {
                "cost_usd": cost,
                "speedup_vs_triton": speedup,
                "kernel": kernel_name
                or _kernel_name_from_path(trace_directory, relative_path),
                "source": str(relative_path),
            }
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


def _scale(value, minimum, maximum, start, span):
    if maximum == minimum:
        return start + span / 2
    return start + (value - minimum) / (maximum - minimum) * span


def write_svg(output_path, frontier):
    width, height = 1100, 700
    left, right, top, bottom = 105, 45, 55, 95
    costs = [point["cost_usd"] for point in frontier]
    speedups = [point["speedup_vs_triton"] for point in frontier]
    minimum_cost, maximum_cost = min(costs), max(costs)
    minimum_speedup, maximum_speedup = min(speedups), max(speedups)
    cost_padding = max((maximum_cost - minimum_cost) * 0.05, 0.01)
    speedup_padding = max((maximum_speedup - minimum_speedup) * 0.08, 0.01)
    minimum_cost = max(0.0, minimum_cost - cost_padding)
    maximum_cost += cost_padding
    minimum_speedup = min(minimum_speedup - speedup_padding, 1.0)
    maximum_speedup = max(maximum_speedup + speedup_padding, 1.0)
    plot_width = width - left - right
    plot_height = height - top - bottom
    kernel_names = sorted({str(point["kernel"]) for point in frontier})
    title_suffix = ", ".join(kernel_names)

    def point_coordinates(point):
        x = _scale(point["cost_usd"], minimum_cost, maximum_cost, left, plot_width)
        y = (
            height
            - bottom
            - _scale(
                point["speedup_vs_triton"],
                minimum_speedup,
                maximum_speedup,
                0,
                plot_height,
            )
        )
        return x, y

    svg = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="700" '
            'viewBox="0 0 1100 700">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="550" y="30" text-anchor="middle" font-family="sans-serif" '
            f'font-size="20">Speedup / cost Pareto frontier - '
            f"{html.escape(title_suffix)}</text>"
        ),
        (
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
            f'y2="{height - bottom}" stroke="black"/>'
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" '
            'stroke="black"/>'
        ),
    ]
    for tick_index in range(6):
        fraction = tick_index / 5
        x = left + fraction * plot_width
        y = height - bottom - fraction * plot_height
        cost = minimum_cost + fraction * (maximum_cost - minimum_cost)
        speedup = minimum_speedup + fraction * (maximum_speedup - minimum_speedup)
        svg.extend(
            [
                (
                    f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
                    f'y2="{height - bottom}" stroke="#dddddd"/>'
                ),
                (
                    f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" '
                    f'y2="{y:.2f}" stroke="#dddddd"/>'
                ),
                (
                    f'<text x="{x:.2f}" y="{height - bottom + 25}" '
                    'text-anchor="middle" font-family="sans-serif" font-size="12">'
                    f"${cost:.3f}</text>"
                ),
                (
                    f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="12">{speedup:.3f}x</text>'
                ),
            ]
        )
    baseline_y = (
        height
        - bottom
        - _scale(
            1.0,
            minimum_speedup,
            maximum_speedup,
            0,
            plot_height,
        )
    )
    svg.extend(
        [
            (
                f'<line x1="{left}" y1="{baseline_y:.2f}" '
                f'x2="{width - right}" y2="{baseline_y:.2f}" '
                'stroke="#2563eb" stroke-dasharray="8 5" stroke-width="2"/>'
            ),
            (
                f'<text x="{width - right - 5}" y="{baseline_y - 6:.2f}" '
                'text-anchor="end" font-family="sans-serif" font-size="12" '
                'fill="#2563eb">1.0x baseline</text>'
            ),
        ]
    )
    coordinates = " ".join(
        f"{x:.2f},{y:.2f}" for x, y in map(point_coordinates, frontier)
    )
    svg.append(
        f'<polyline points="{coordinates}" fill="none" stroke="#dc2626" '
        'stroke-width="3"/>'
    )
    for point in frontier:
        x, y = point_coordinates(point)
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="#dc2626"/>')
    svg.extend(
        [
            (
                f'<text x="550" y="{height - 25}" text-anchor="middle" '
                'font-family="sans-serif" font-size="16">Cumulative API cost (USD)</text>'
            ),
            (
                '<text x="25" y="350" text-anchor="middle" '
                'font-family="sans-serif" font-size="16" '
                'transform="rotate(-90 25 350)">Speedup vs. Triton</text>'
            ),
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Plot the speedup/cost Pareto frontier from agent trace JSON."
    )
    parser.add_argument("trace_directory", type=Path)
    arguments = parser.parse_args()
    trace_directory = arguments.trace_directory.resolve()
    if not trace_directory.is_dir():
        parser.error(f"Not a directory: {trace_directory}")

    accepted_kernels = load_accepted_kernels(trace_directory)
    if not accepted_kernels:
        raise RuntimeError("No accepted kernels with run_cost_usd_so_far were found.")
    frontier = pareto_frontier(accepted_kernels)
    plots_directory = Path(__file__).resolve().parent / "plots"
    plots_directory.mkdir(exist_ok=True)
    svg_path = plots_directory / PLOT_FILENAME
    write_svg(svg_path, frontier)
    print(f"Saved {len(frontier)} frontier points to {svg_path}")


if __name__ == "__main__":
    main()
