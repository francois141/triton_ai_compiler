import json
import re
from pathlib import Path

from prompts import build_prompt_sections_for_operator
from triton_ptx import (
    dump_kernel_ptx,
    get_ptx_system_config,
    parse_ptx_signature,
    resolve_kernel,
)
from utils.response_format import PtxKernel

TRITON_GENERATED_PTX_DIRECTORY = (
    Path(__file__).resolve().parent.parent / "triton_generated_ptx"
)

_REQNTID_PATTERN = re.compile(
    r"^\s*\.reqntid\s+(\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?\s*$",
    re.MULTILINE,
)


def build_openai_tools(skill_ids=None):
    tools = [
        {
            "type": "function",
            "name": "launch_verifier",
            "description": (
                "Compile, verify, and benchmark a PTX candidate. Use this "
                "before returning any candidate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ptx": {"type": "string", "minLength": 1},
                    "num_threads_x": {"type": "integer", "minimum": 1},
                    "num_threads_y": {"type": "integer", "minimum": 1},
                    "num_threads_z": {"type": "integer", "minimum": 1},
                },
                "required": [
                    "ptx",
                    "num_threads_x",
                    "num_threads_y",
                    "num_threads_z",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]
    tools.extend(
        [
            {
                "type": "function",
                "name": "apply_ptx_patch",
                "description": (
                    "Apply a unified diff to the current candidate PTX file. "
                    "Use this to make every source-code change; do not return "
                    "complete PTX in the final response."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patch": {"type": "string", "minLength": 1},
                        "num_threads_x": {"type": "integer", "minimum": 1},
                        "num_threads_y": {"type": "integer", "minimum": 1},
                        "num_threads_z": {"type": "integer", "minimum": 1},
                    },
                    "required": [
                        "patch",
                        "num_threads_x",
                        "num_threads_y",
                        "num_threads_z",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "verify_current_ptx",
                "description": (
                    "Compile, verify, and benchmark the current PTX file after "
                    "applying a patch."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]
    )
    if skill_ids:
        if isinstance(skill_ids, str):
            skill_ids = [skill_ids]
        tools.append(
            {
                "type": "shell",
                "environment": {
                    "type": "container_auto",
                    "skills": [
                        {
                            "type": "skill_reference",
                            "skill_id": skill_id,
                            "version": "latest",
                        }
                        for skill_id in skill_ids
                    ],
                },
            }
        )
    return tools


def build_anthropic_tools():
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }
        for tool in build_openai_tools()
        if tool["type"] == "function"
    ]


def build_prompt_sections(kernel_name, kernel=None, *, enable_web_search=True):
    kernel = resolve_kernel(kernel_name)() if kernel is None else kernel
    version, target, address_size = get_ptx_system_config()
    signature = parse_ptx_signature(dump_kernel_ptx(kernel))
    return build_prompt_sections_for_operator(
        kernel,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=signature,
        enable_web_search=enable_web_search,
    )


def load_start_json(start_json):
    candidate, _ = load_start_json_with_autotune(start_json)
    return candidate


def load_start_json_with_autotune(start_json):
    if start_json is None:
        return None, None
    value = str(start_json)
    serialized_candidate = (
        value
        if value.lstrip().startswith("{")
        else Path(value).read_text(encoding="utf-8")
    )
    loaded_data = json.loads(serialized_candidate)
    if not isinstance(loaded_data, dict):
        raise ValueError("Starting candidate JSON must contain an object.")
    candidate_data = loaded_data.get(
        "payload",
        loaded_data.get(
            "candidate",
            loaded_data.get("resulting_payload", loaded_data),
        ),
    )
    if not isinstance(candidate_data, dict):
        raise ValueError("Starting candidate payload must contain an object.")
    autotune_metrics = loaded_data.get("autotune_metrics")
    if autotune_metrics is None:
        autotune_metrics = candidate_data.get("autotune_metrics")
    candidate_data = {
        key: value
        for key, value in candidate_data.items()
        if key in PtxKernel.model_fields
    }
    if autotune_metrics is not None and not isinstance(autotune_metrics, dict):
        raise ValueError("autotune_metrics must contain an object.")
    return PtxKernel.model_validate(candidate_data), autotune_metrics


def load_start_ptx(start_ptx, *, num_threads_x, num_threads_y, num_threads_z):
    if start_ptx is None:
        return None
    return PtxKernel(
        ptx=Path(start_ptx).read_text(encoding="utf-8"),
        num_threads_x=num_threads_x,
        num_threads_y=num_threads_y,
        num_threads_z=num_threads_z,
    )


def load_triton_generated_ptx(kernel_name):
    ptx_path = TRITON_GENERATED_PTX_DIRECTORY / f"{kernel_name}.ptx"
    try:
        ptx = ptx_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No generated PTX found for {kernel_name!r}: {ptx_path}"
        ) from exc

    required_threads = _REQNTID_PATTERN.search(ptx)
    if required_threads is None:
        return PtxKernel(ptx=ptx, num_threads_x=128)

    threads_x, threads_y, threads_z = required_threads.groups()
    return PtxKernel(
        ptx=ptx,
        num_threads_x=int(threads_x),
        num_threads_y=int(threads_y) if threads_y is not None else 1,
        num_threads_z=int(threads_z) if threads_z is not None else 1,
    )
