from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


PositiveInteger = Annotated[int, Field(ge=1)]


class PtxKernel(BaseModel):
    """Represent a validated PTX kernel returned by an LLM."""

    model_config = ConfigDict(extra="forbid")

    ptx: str
    num_threads_x: PositiveInteger
    num_threads_y: PositiveInteger = 1
    num_threads_z: PositiveInteger = 1


PTX_KERNEL_JSON_SCHEMA = PtxKernel.model_json_schema()
PTX_KERNEL_JSON_SCHEMA["required"] = list(
    PTX_KERNEL_JSON_SCHEMA["properties"]
)

PTX_KERNEL_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "ptx_kernel",
    "strict": True,
    "schema": PTX_KERNEL_JSON_SCHEMA,
}

IMPROVEMENT_PLAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "improvement_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "improvements": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "rationale": {"type": "string", "minLength": 1},
                        "instruction": {"type": "string", "minLength": 1},
                    },
                    "required": ["name", "rationale", "instruction"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["improvements"],
        "additionalProperties": False,
    },
}
