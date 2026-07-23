import json
from pathlib import Path

from prompts import build_prompt_for_operator
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


def build_openai_tools(skill_id=None):
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
    if skill_id is not None:
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
                    ],
                },
            }
        )
    return tools


def build_anthropic_tools():
    verifier_tool = build_openai_tools()[0]
    return [
        {
            "name": verifier_tool["name"],
            "description": verifier_tool["description"],
            "input_schema": verifier_tool["parameters"],
        }
    ]


def build_initial_prompt(kernel_name):
    kernel_cls = resolve_kernel(kernel_name)
    version, target, address_size = get_ptx_system_config()
    signature = parse_ptx_signature(dump_kernel_ptx(kernel_cls()))
    return build_prompt_for_operator(
        kernel_cls,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=signature,
    )


def load_start_json(start_json):
    if start_json is None:
        return None
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
        "payload", loaded_data.get("candidate", loaded_data)
    )
    if not isinstance(candidate_data, dict):
        raise ValueError("Starting candidate payload must contain an object.")
    candidate_data = {
        key: value
        for key, value in candidate_data.items()
        if key in PtxKernel.model_fields
    }
    return PtxKernel.model_validate(candidate_data)


def load_triton_generated_ptx(kernel_name):
    ptx_path = TRITON_GENERATED_PTX_DIRECTORY / f"{kernel_name}.ptx"
    try:
        ptx = ptx_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No generated PTX found for {kernel_name!r}: {ptx_path}"
        ) from exc

    return PtxKernel(ptx=ptx, num_threads_x=128)
