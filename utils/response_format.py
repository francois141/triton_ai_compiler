from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

PositiveInteger = Annotated[int, Field(ge=1)]


class PtxKernel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ptx: str = Field(min_length=1)
    num_threads_x: PositiveInteger
    num_threads_y: PositiveInteger = 1
    num_threads_z: PositiveInteger = 1
    difficulties: list[str] = Field(default_factory=list, max_length=3)

    @field_validator("ptx")
    @classmethod
    def validate_ptx(cls, value):
        if not value.strip():
            raise ValueError('"ptx" must contain non-whitespace PTX code.')
        return value


class PtxKernelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_threads_x: PositiveInteger
    num_threads_y: PositiveInteger = 1
    num_threads_z: PositiveInteger = 1
    difficulties: list[str] = Field(default_factory=list, max_length=3)


class FailureAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_cause: str = Field(min_length=1)
    repair_instruction: str = Field(min_length=1)


PTX_KERNEL_JSON_SCHEMA = PtxKernel.model_json_schema()
PTX_KERNEL_JSON_SCHEMA["required"] = list(PTX_KERNEL_JSON_SCHEMA["properties"])

PTX_KERNEL_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "ptx_kernel",
    "strict": True,
    "schema": PTX_KERNEL_JSON_SCHEMA,
}

PTX_KERNEL_METADATA_JSON_SCHEMA = PtxKernelMetadata.model_json_schema()
PTX_KERNEL_METADATA_JSON_SCHEMA["required"] = list(
    PTX_KERNEL_METADATA_JSON_SCHEMA["properties"]
)

PTX_KERNEL_METADATA_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "ptx_kernel_metadata",
    "strict": True,
    "schema": PTX_KERNEL_METADATA_JSON_SCHEMA,
}

FAILURE_ANALYSIS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "failure_analysis",
    "strict": True,
    "schema": FailureAnalysis.model_json_schema(),
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
                "minItems": 1,
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
