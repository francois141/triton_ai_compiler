import argparse
import json
from pathlib import Path

from utils.evaluation import normalize_nested_json
from utils.response import verifier_for_kernel


def load_candidate_record(path):
    print("Loading the JSON")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")

    print("Json loaded")

    return data


def extract_payload(record):
    payload = record.get("payload")

    if isinstance(payload, dict):
        return payload

    candidate = record.get("candidate")
    if isinstance(candidate, dict) and "ptx" in candidate:
        return candidate

    evaluation = record.get("evaluation")
    if isinstance(evaluation, dict):
        evaluated_payload = evaluation.get("payload")
        if isinstance(evaluated_payload, dict):
            return evaluated_payload

    # Accept a raw payload file too, so users can remeasure either archived
    # EvaluatedCandidate JSON or just {"ptx": ..., "num_threads_x": ...}.
    if "ptx" in record:
        return record

    raise ValueError(
        'Input JSON must contain a "payload" or "candidate" PTX object, or be '
        "a raw PTX payload."
    )


def extract_autotune_metrics(record):
    autotune_metrics = record.get("autotune_metrics")
    if autotune_metrics is not None:
        return autotune_metrics

    for payload_key in ("payload", "candidate", "resulting_payload"):
        payload = record.get(payload_key)
        if isinstance(payload, dict) and "autotune_metrics" in payload:
            return payload["autotune_metrics"]

    return None


def resolve_kernel_name(record, explicit_kernel):
    evaluation = record.get("evaluation")
    archived_kernel_name = (
        evaluation.get("kernel_name") if isinstance(evaluation, dict) else None
    )
    kernel_name = explicit_kernel or record.get("kernel_name") or archived_kernel_name
    if not isinstance(kernel_name, str) or not kernel_name.strip():
        raise ValueError(
            'Kernel name is missing. Include "kernel_name" in the JSON or pass '
            "--kernel."
        )
    return kernel_name


def remeasure_candidate(record, *, kernel_name=None):
    from ptx_gym.evaluation import Payload

    resolved_kernel_name = resolve_kernel_name(record, kernel_name)
    payload = Payload.from_input(extract_payload(record))
    evaluator = verifier_for_kernel(
        resolved_kernel_name,
        extract_autotune_metrics(record),
    )
    return evaluator.evaluate(payload)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-run compile, correctness, and timing measurement for an archived "
            "PTX candidate JSON."
        )
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to an evaluated candidate, archived trace, or raw PTX payload JSON.",
    )
    parser.add_argument(
        "--kernel",
        help='Kernel class name to use when the input JSON has no "kernel_name".',
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the fresh measurement JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    record = load_candidate_record(args.json_path)
    result = remeasure_candidate(record, kernel_name=args.kernel)
    output_data = json.loads(result.to_json())
    output_data.pop("ncu_report", None)
    output = json.dumps(
        normalize_nested_json(output_data),
        indent=2,
        ensure_ascii=False,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")

    print(output)


if __name__ == "__main__":
    main()
