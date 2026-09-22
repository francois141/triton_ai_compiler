import argparse
import json
import logging
import re
from collections import namedtuple
from pathlib import Path

LOGGER = logging.getLogger(__name__)
SPEEDUP_PATTERN = re.compile(r"speedup_vs_triton_(?P<speedup>[0-9]+(?:\.[0-9]+)?)x")
PRECISION_PATTERN = re.compile(
    r"(?P<base>.+?)(?P<precision>Float(?:8|16|32|64)|BFloat16)"
    r"(?P<suffix>Kernel)?$"
)
MISSING_GPU_TYPE = "NVIDIA L40S"


Candidate = namedtuple(
    "Candidate",
    (
        "kernel_dir",
        "kernel_name",
        "gpu",
        "precision",
        "speedup",
        "ptx",
        "source",
        "hyperparameters",
    ),
)


def parse_kernel_name(directory_name):
    parts = directory_name.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse kernel directory name: {directory_name}")

    run_name = parts[1]
    model_marker = re.search(r"_(?:gpt|claude)-", run_name)
    if model_marker is None:
        raise ValueError(f"Cannot find model marker in: {directory_name}")

    kernel_name = run_name[: model_marker.start()]
    match = PRECISION_PATTERN.fullmatch(kernel_name)
    if match is None:
        return kernel_name, "Unknown"

    return (
        match.group("base") + (match.group("suffix") or ""),
        match.group("precision"),
    )


def load_gpu_type(run_directory):
    metadata_path = run_directory / "events_speedup_vs_triton_pending.json"
    if not metadata_path.is_file():
        return None

    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not read %s: %s", metadata_path, exc)
        return None

    if isinstance(metadata, dict):
        return metadata.get("gpu_type")
    if isinstance(metadata, list):
        for event in metadata:
            if isinstance(event, dict) and event.get("gpu_type"):
                return event["gpu_type"]
    return None


def parse_speedup(path, data):
    speedup = data.get("speedup")
    if isinstance(speedup, (int, float)):
        return float(speedup)

    match = SPEEDUP_PATTERN.search(path.name)
    return float(match.group("speedup")) if match else None


def candidate_from_json(path, data):
    candidate = data.get("candidate")
    evaluation = data.get("evaluation")
    if isinstance(candidate, dict) and isinstance(evaluation, dict):
        return candidate.get("ptx"), evaluation.get("speedup_vs_triton")
    return data.get("ptx"), parse_speedup(path, data)


def extract_hyperparameters(data):
    threads_source = data.get("ptx_hyperparameters") or data.get("candidate") or data
    hyperparameters = {}
    if isinstance(threads_source, dict):
        for key in ("num_threads_x", "num_threads_y", "num_threads_z"):
            if key in threads_source:
                hyperparameters[key] = threads_source[key]

    autotune_metrics = data.get("autotune_metrics")
    selected_config = (
        autotune_metrics.get("selected_config")
        if isinstance(autotune_metrics, dict)
        else None
    )
    if isinstance(selected_config, dict):
        for key in ("block_m", "block_n", "block_k", "num_warps"):
            if key in selected_config:
                hyperparameters[key] = selected_config[key]

    return hyperparameters


def tool_output_metadata_path(ptx_path):
    stem, _, _ = ptx_path.stem.partition("_speedup_vs_triton_")
    return ptx_path.with_name(f"{stem}.json")


def tool_output_speedup(metadata_path, data):
    speedup = data.get("speedup_vs_triton")
    if isinstance(speedup, (int, float)):
        return float(speedup)

    answer = data.get("answer")
    if isinstance(answer, dict):
        speedup = answer.get("speedup_vs_triton")
        if isinstance(speedup, (int, float)):
            return float(speedup)
    return parse_speedup(metadata_path, data)


def append_candidate(
    candidates,
    run_directory,
    kernel_name,
    precision,
    gpu,
    speedup,
    ptx,
    source,
    hyperparameters,
):
    if not isinstance(ptx, str) or not ptx.strip() or speedup is None:
        return
    candidate_gpu = gpu or MISSING_GPU_TYPE
    if not isinstance(candidate_gpu, str) or not candidate_gpu.strip():
        return
    candidates.append(
        Candidate(
            kernel_dir=run_directory.name,
            kernel_name=kernel_name,
            gpu=candidate_gpu,
            precision=precision,
            speedup=speedup,
            ptx=ptx,
            source=source,
            hyperparameters=hyperparameters,
        )
    )


def collect_candidates(astra_dir):
    candidates = []
    for run_directory in sorted(path for path in astra_dir.iterdir() if path.is_dir()):
        try:
            kernel_name, precision = parse_kernel_name(run_directory.name)
        except ValueError as exc:
            LOGGER.warning("Skipping %s: %s", run_directory, exc)
            continue

        gpu = load_gpu_type(run_directory)

        for json_path in sorted(run_directory.rglob("*.json")):
            try:
                data = json.loads(json_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.warning("Could not read %s: %s", json_path, exc)
                continue

            if not isinstance(data, dict):
                continue

            ptx, speedup = candidate_from_json(json_path, data)
            append_candidate(
                candidates,
                run_directory,
                kernel_name,
                precision,
                data.get("gpu_type") or gpu,
                speedup,
                ptx,
                json_path,
                extract_hyperparameters(data),
            )

        for ptx_path in sorted(
            (run_directory / "tool_output").glob("*_speedup_vs_triton_*.ptx")
        ):
            metadata_path = tool_output_metadata_path(ptx_path)
            try:
                metadata = json.loads(metadata_path.read_text())
                ptx = ptx_path.read_text()
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.warning("Could not read tool output %s: %s", ptx_path, exc)
                continue
            if not isinstance(metadata, dict):
                continue

            append_candidate(
                candidates,
                run_directory,
                kernel_name,
                precision,
                metadata.get("gpu_type") or gpu,
                tool_output_speedup(metadata_path, metadata),
                ptx,
                ptx_path,
                extract_hyperparameters(metadata),
            )

    return candidates


def select_best(candidates):
    best = {}
    for candidate in candidates:
        key = (candidate.kernel_name, candidate.gpu, candidate.precision)
        current = best.get(key)
        if current is None or (candidate.speedup, candidate.source.as_posix()) > (
            current.speedup,
            current.source.as_posix(),
        ):
            best[key] = candidate
    return best


def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9.-]+", "_", value).strip("._")


def write_best_ptx(best, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for output_path in output_dir.rglob("*.ptx"):
        output_path.unlink()
    for sidecar_path in output_dir.rglob("*.json"):
        sidecar_path.unlink()
    for candidate in sorted(best.values(), key=lambda item: item.source.as_posix()):
        kernel_dir = output_dir / safe_filename(candidate.kernel_name)
        stem = (
            f"{safe_filename(candidate.gpu)}_{candidate.precision}_"
            f"{candidate.speedup:.4f}x"
        )
        kernel_dir.mkdir(parents=True, exist_ok=True)
        output_path = kernel_dir / f"{stem}.ptx"
        output_path.write_text(candidate.ptx)
        if candidate.hyperparameters:
            (kernel_dir / f"{stem}.json").write_text(
                json.dumps(candidate.hyperparameters, indent=2)
            )
        LOGGER.info(
            "Wrote %.4fx %s to %s", candidate.speedup, candidate.source, output_path
        )


def main(astra_dir, output_dir):
    candidates = collect_candidates(astra_dir)
    best = select_best(candidates)
    write_best_ptx(best, output_dir)
    LOGGER.info("Wrote %d PTX files from %d candidates", len(best), len(candidates))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract the fastest PTX candidate from an Astra run directory."
    )
    parser.add_argument(
        "astra_dir",
        nargs="?",
        type=Path,
        default=Path("astra"),
        help="Directory containing Astra kernel run directories.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("final_ptx"),
        help="Directory where selected PTX files are written.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    if not args.astra_dir.is_dir():
        parser.error(f"Astra directory does not exist: {args.astra_dir}")
    main(args.astra_dir, args.output_dir)
