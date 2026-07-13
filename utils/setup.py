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


def build_tools(skill_id):
    if skill_id is None:
        return []
    return [
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
    candidate_data = loaded_data.get("payload", loaded_data)
    if not isinstance(candidate_data, dict):
        raise ValueError("Starting candidate payload must contain an object.")
    candidate_data = {
        key: value
        for key, value in candidate_data.items()
        if key in PtxKernel.model_fields
    }
    return PtxKernel.model_validate(candidate_data)
