import json
from pathlib import Path

from prompts import build_prompt_for_kernel
from triton_api import get_kernel_data
from utils.response_format import PtxKernel


def build_tools(skill_id):
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


def build_initial_prompt(kernel_name):
    return build_prompt_for_kernel(get_kernel_data(kernel_name))


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
    candidate_data = loaded_data.get("payload", loaded_data.get("candidate", loaded_data))
    if not isinstance(candidate_data, dict):
        raise ValueError("Starting candidate payload must contain an object.")
    candidate_data = {
        key: value
        for key, value in candidate_data.items()
        if key in PtxKernel.model_fields
    }
    return PtxKernel.model_validate(candidate_data)
