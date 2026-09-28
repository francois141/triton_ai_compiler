import argparse
import json
import logging
import math
from pathlib import Path
from statistics import median

LOGGER = logging.getLogger(__name__)
CORRECTION_FACTOR_FILENAME = "correction factor.txt"


def _positive_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def _evaluation_record(data):
    evaluation = data.get("evaluation")
    return evaluation if isinstance(evaluation, dict) else data


def archived_baseline(run_directory):
    kernel_names = set()
    triton_p50_values = []
    for json_path in sorted(run_directory.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            LOGGER.warning("Skipping unreadable JSON %s: %s", json_path, error)
            continue
        if not isinstance(data, dict):
            continue

        evaluation = _evaluation_record(data)
        triton_p50 = _positive_number(evaluation.get("triton_p50"))
        kernel_name = evaluation.get("kernel_name")
        if triton_p50 is None or not isinstance(kernel_name, str):
            continue
        kernel_names.add(kernel_name)
        triton_p50_values.append(triton_p50)

    if not triton_p50_values:
        raise ValueError("No archived Triton p50 was found")
    if len(kernel_names) != 1:
        names = ", ".join(sorted(kernel_names))
        raise ValueError(f"Expected one kernel name, found: {names}")
    return kernel_names.pop(), median(triton_p50_values)


def measure_triton_p50(kernel_name):
    import torch
    from ptx_gym.evaluation.performance import benchmark
    from ptx_gym.kernels import resolve_kernel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to remeasure a Triton baseline.")
    kernel = resolve_kernel(kernel_name)()
    inputs = kernel.get_random_input()
    return benchmark(lambda: kernel.forward_triton(inputs)).p50


def remeasure_run(run_directory, *, write=True):
    kernel_name, archived_p50 = archived_baseline(run_directory)
    fresh_p50 = measure_triton_p50(kernel_name)
    correction_factor = fresh_p50 / archived_p50
    correction_path = run_directory / CORRECTION_FACTOR_FILENAME
    if write:
        correction_path.write_text(f"{correction_factor:.17g}\n", encoding="utf-8")
    return kernel_name, archived_p50, fresh_p50, correction_factor


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Remeasure Triton baselines for trace runs and write the correction "
            "factors used by plot_speedup_cost_frontier.py."
        )
    )
    parser.add_argument(
        "--trace-directory",
        type=Path,
        default=Path("astra"),
        help="Directory whose immediate subdirectories are trace runs (default: astra).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Measure and report factors without writing correction factor.txt files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.trace_directory.is_dir():
        LOGGER.error("Trace directory does not exist: %s", args.trace_directory)
        return 1

    failures = 0
    for run_directory in sorted(path for path in args.trace_directory.iterdir() if path.is_dir()):
        try:
            kernel_name, archived_p50, fresh_p50, correction_factor = remeasure_run(
                run_directory,
                write=not args.dry_run,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            failures += 1
            LOGGER.error("Failed to remeasure %s: %s", run_directory.name, error)
            continue
        print(
            f"{run_directory.name}: {kernel_name}, archived p50={archived_p50:.6f} "
            f"ms, fresh p50={fresh_p50:.6f} ms, factor={correction_factor:.6f}"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
