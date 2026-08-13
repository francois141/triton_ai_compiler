import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from prompts.blocks import system_prompt
from triton_ptx import Payload, TritonPTXCandidateEvaluator, resolve_kernel

from .cost import COST_LOG_PATH, append_cost_log, append_pipeline_cost_summary
from .response_format import PtxKernel
from .traces import record_tool_call

RESPONSE_RETRY_ATTEMPTS = 6
ANTHROPIC_MAX_TOKENS = 16_384
ANTHROPIC_SKILLS_BETA = "skills-2025-10-02"
ANTHROPIC_CODE_EXECUTION_TOOL = {
    "type": "code_execution_20260521",
    "name": "code_execution",
}
PATCH_WORKFLOW_TOOL_NAMES = frozenset({"apply_ptx_patch", "verify_current_ptx"})
FLOAT16_GEMM_WEB_SEARCH_TOOL = {
    "type": "web_search",
    "filters": {
        "allowed_domains": [
            "github.com",
            "leimao.github.io",
            "docs.nvidia.com",
            "www.rimikawrites.com",
            "qsysarch.com",
            "siboehm.com",
            "alexarmbr.github.io",
            "hazyresearch.stanford.edu",
        ]
    },
}


@cache
def _default_verifier_for_kernel(kernel_name):
    return TritonPTXCandidateEvaluator(resolve_kernel(kernel_name))


def verifier_for_kernel(kernel_name, autotune_metrics=None):
    if autotune_metrics is None:
        return _default_verifier_for_kernel(kernel_name)

    selected_config = autotune_metrics.get("selected_config")
    if not isinstance(selected_config, dict):
        raise ValueError("autotune_metrics must include a selected_config object.")
    if not all(
        isinstance(name, str) and isinstance(value, int)
        for name, value in selected_config.items()
    ):
        raise ValueError("autotune_metrics selected_config must map names to integers.")

    operator_cls = resolve_kernel(kernel_name)
    operator = operator_cls(ptx={"tuning_config": selected_config})
    operator.tuning_result = autotune_metrics.get("tuning_result")
    return TritonPTXCandidateEvaluator(operator_cls, operator=operator)


def _get_field(value, field_name):
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _tools_for_request(tools, current_candidate):
    if current_candidate is not None:
        return tools
    return [tool for tool in tools if tool.get("name") not in PATCH_WORKFLOW_TOOL_NAMES]


def _format_tool_call(output_item):
    item_type = _get_field(output_item, "type")
    if not isinstance(item_type, str) or not item_type.endswith("_call"):
        return None

    tool_name = (
        _get_field(output_item, "name")
        or _get_field(output_item, "tool_name")
        or _get_field(output_item, "server_label")
        or item_type.removesuffix("_call")
    )
    call_id = _get_field(output_item, "call_id") or _get_field(output_item, "id")
    status = _get_field(output_item, "status")
    details = [f"tool={tool_name}"]
    if call_id is not None:
        details.append(f"id={call_id}")
    if status is not None:
        details.append(f"status={status}")
    return ", ".join(details)


def print_tool_calls(response):
    output_items = _get_field(response, "output")
    if output_items is None:
        return
    for output_item in output_items:
        tool_call = _format_tool_call(output_item)
        if tool_call is not None:
            print(f"=== LLM called tool: {tool_call} ===", flush=True)


def _append_response_cost(*, model, response, cost_log_path, pipeline):
    cost = append_cost_log(
        model=model,
        response=response,
        cost_log_path=COST_LOG_PATH,
        pipeline=pipeline,
    )
    if cost_log_path is not None and Path(cost_log_path) != COST_LOG_PATH:
        append_cost_log(
            model=model,
            response=response,
            cost_log_path=cost_log_path,
            pipeline=pipeline,
        )
    return cost


def _append_pipeline_cost(*, pipeline, cost, cost_log_path):
    append_pipeline_cost_summary(
        pipeline=pipeline,
        cost=cost,
        cost_log_path=COST_LOG_PATH,
    )
    if Path(cost_log_path) != COST_LOG_PATH:
        append_pipeline_cost_summary(
            pipeline=pipeline,
            cost=cost,
            cost_log_path=cost_log_path,
        )


def _message_output_text(output_item):
    if _get_field(output_item, "type") != "message":
        return None
    content_items = _get_field(output_item, "content")
    if content_items is None:
        return None
    text_parts = [
        text
        for content_item in content_items
        if _get_field(content_item, "type") == "output_text"
        for text in [_get_field(content_item, "text")]
        if isinstance(text, str)
    ]
    return "".join(text_parts) if text_parts else None


def response_json_text(response):
    output_items = _get_field(response, "output")
    if output_items is None:
        output_text = _get_field(response, "output_text")
        if isinstance(output_text, str) and output_text:
            return output_text
        raise ValueError("Response did not contain output text.")

    fallback_text = None
    for output_item in output_items:
        output_text = _message_output_text(output_item)
        if output_text is None:
            continue
        if _get_field(output_item, "phase") == "final_answer":
            return output_text
        fallback_text = output_text
    if fallback_text is None:
        raise ValueError("Response did not contain message output text.")
    return fallback_text


@dataclass(slots=True)
class _AnthropicResponse:
    output_text: str
    usage: object
    response: object

    def model_dump(self, mode="json"):
        del mode
        return {
            "provider": "anthropic",
            "output_text": self.output_text,
            "response": _json_value(self.response),
        }


def _json_value(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _anthropic_response_tool(response_format):
    return {
        "name": "submit_response",
        "description": (
            "Submit the final response after using launch_verifier as needed. "
            "The submitted value must exactly match this response schema."
        ),
        "input_schema": response_format["schema"],
    }


class PtxPatchWorkspace:
    def __init__(self, candidate):
        self.candidate = candidate

    def apply_patch(self, arguments):
        patch = arguments["patch"]
        headers = [
            line for line in patch.splitlines() if line.startswith(("--- ", "+++ "))
        ]
        if len(headers) != 2 or headers != ["--- candidate.ptx", "+++ candidate.ptx"]:
            raise ValueError(
                "PTX patch must modify only candidate.ptx with standard unified "
                "diff headers."
            )
        with tempfile.TemporaryDirectory() as directory:
            ptx_path = Path(directory) / "candidate.ptx"
            ptx_path.write_text(self.candidate.ptx, encoding="utf-8")
            result = subprocess.run(
                ["patch", "--batch", "--forward", "--strip=0", "--input=-"],
                cwd=directory,
                input=patch,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                output = (result.stdout + result.stderr).strip()
                raise ValueError(f"PTX patch did not apply: {output}")
            ptx = ptx_path.read_text(encoding="utf-8")

        self.candidate = PtxKernel(
            ptx=ptx,
            num_threads_x=arguments["num_threads_x"],
            num_threads_y=arguments["num_threads_y"],
            num_threads_z=arguments["num_threads_z"],
        )
        return {
            "status": "applied",
            "ptx_lines": self.candidate.ptx.count("\n") + 1,
            "num_threads_x": self.candidate.num_threads_x,
            "num_threads_y": self.candidate.num_threads_y,
            "num_threads_z": self.candidate.num_threads_z,
        }


def request_anthropic_json(
    client,
    *,
    model,
    prompt,
    response_format,
    tools,
    kernel_name,
    skill_ids,
    cost_log_path=None,
    pipeline=None,
    current_candidate=None,
    autotune_metrics=None,
    system_instruction=None,
):
    verifier = verifier_for_kernel(kernel_name, autotune_metrics)
    workspace = (
        PtxPatchWorkspace(current_candidate) if current_candidate is not None else None
    )
    available_tools = _tools_for_request(tools, current_candidate)
    messages = [{"role": "user", "content": prompt}]
    anthropic_tools = [
        *available_tools,
        _anthropic_response_tool(response_format),
        ANTHROPIC_CODE_EXECUTION_TOOL,
    ]
    container = {
        "skills": [
            {
                "type": "custom",
                "skill_id": skill_id,
                "version": "latest",
            }
            for skill_id in skill_ids
        ]
    }
    total_cost = None

    while True:
        response = client.beta.messages.create(
            model=model,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=system_prompt()
            if system_instruction is None
            else system_instruction,
            messages=messages,
            tools=anthropic_tools,
            container=container,
            betas=[ANTHROPIC_SKILLS_BETA],
        )
        cost = _append_response_cost(
            model=model,
            response=response,
            cost_log_path=cost_log_path,
            pipeline=pipeline,
        )
        if cost is not None:
            total_cost = (total_cost or 0) + cost

        tool_uses = [
            block
            for block in response.content
            if _get_field(block, "type") == "tool_use"
        ]
        submitted_response = next(
            (
                tool_use
                for tool_use in tool_uses
                if _get_field(tool_use, "name") == "submit_response"
            ),
            None,
        )
        if submitted_response is not None:
            response_text = json.dumps(_get_field(submitted_response, "input"))
            if cost_log_path is not None:
                _append_pipeline_cost(
                    pipeline=pipeline or "unnamed",
                    cost=total_cost,
                    cost_log_path=cost_log_path,
                )
            return (
                _AnthropicResponse(response_text, response.usage, response),
                total_cost,
                workspace.candidate if workspace is not None else None,
            )
        if _get_field(response, "stop_reason") == "pause_turn":
            container_id = _get_field(_get_field(response, "container"), "id")
            if isinstance(container_id, str):
                container["id"] = container_id
            messages.append(
                {
                    "role": "assistant",
                    "content": [_json_value(block) for block in response.content],
                }
            )
            continue
        if not tool_uses:
            raise ValueError(
                "Anthropic response did not call submit_response with the "
                "required structured output."
            )

        tool_results = []
        for tool_use in tool_uses:
            tool_name = _get_field(tool_use, "name")
            tool_id = _get_field(tool_use, "id")
            resulting_payload = None
            print(
                f"=== LLM called tool: tool={tool_name}, id={tool_id} ===",
                flush=True,
            )
            arguments = _get_field(tool_use, "input")
            if tool_name == "launch_verifier":
                evaluation = verifier.evaluate(Payload.from_input(arguments))
                result = evaluation.to_llm()
            elif tool_name == "apply_ptx_patch" and workspace is not None:
                try:
                    result = json.dumps(workspace.apply_patch(arguments))
                    resulting_payload = workspace.candidate.model_dump(
                        exclude_none=False
                    )
                except ValueError as error:
                    result = json.dumps({"error": str(error)})
            elif tool_name == "verify_current_ptx" and workspace is not None:
                evaluation = verifier.evaluate(
                    Payload.from_input(workspace.candidate.model_dump())
                )
                result = evaluation.to_llm()
            else:
                raise RuntimeError(f"Unsupported Anthropic tool call: {tool_name}")
            if cost_log_path is not None:
                record_tool_call(
                    Path(cost_log_path).parent,
                    provider="anthropic",
                    tool_name=tool_name,
                    call_id=tool_id,
                    payload=arguments,
                    answer=result,
                    resulting_payload=resulting_payload,
                )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result,
                }
            )
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [_json_value(block) for block in response.content],
                },
                {"role": "user", "content": tool_results},
            ]
        )


def request_openai_json(
    client,
    *,
    model,
    prompt,
    response_format,
    reasoning_effort,
    tools,
    kernel_name,
    cost_log_path=None,
    pipeline=None,
    current_candidate=None,
    autotune_metrics=None,
    system_instruction=None,
):
    from openai import NotFoundError

    verifier = verifier_for_kernel(kernel_name, autotune_metrics)
    workspace = (
        PtxPatchWorkspace(current_candidate) if current_candidate is not None else None
    )
    available_tools = _tools_for_request(tools, current_candidate)
    if kernel_name == "MatrixMultiplicationFloat16":
        available_tools = [*available_tools, FLOAT16_GEMM_WEB_SEARCH_TOOL]
    kwargs = {
        "model": model,
        "instructions": (
            system_prompt() if system_instruction is None else system_instruction
        ),
        "input": [{"role": "user", "content": prompt}],
        "tools": available_tools,
        "text": {"format": response_format},
    }
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    total_cost = None
    while True:
        for retry_index in range(RESPONSE_RETRY_ATTEMPTS):
            try:
                response = client.responses.create(**kwargs)
                break
            except NotFoundError as error:
                if "Skill version" not in str(error):
                    raise
                if retry_index == RESPONSE_RETRY_ATTEMPTS - 1:
                    raise
                delay_seconds = 2**retry_index
                print(
                    "=== Uploaded skill version is not available yet; retrying "
                    f"in {delay_seconds}s ===",
                    flush=True,
                )
                time.sleep(delay_seconds)

        print_tool_calls(response)
        cost = _append_response_cost(
            model=model,
            response=response,
            cost_log_path=cost_log_path,
            pipeline=pipeline,
        )
        if cost is not None:
            total_cost = (total_cost or 0) + cost

        function_calls = [
            item
            for item in (_get_field(response, "output") or [])
            if _get_field(item, "type") == "function_call"
        ]
        if not function_calls:
            if cost_log_path is not None:
                _append_pipeline_cost(
                    pipeline=pipeline or "unnamed",
                    cost=total_cost,
                    cost_log_path=cost_log_path,
                )
            return response, total_cost, workspace.candidate if workspace else None

        tool_outputs = []
        for function_call in function_calls:
            tool_name = _get_field(function_call, "name")
            arguments = json.loads(_get_field(function_call, "arguments"))
            resulting_payload = None
            if tool_name == "launch_verifier":
                evaluation = verifier.evaluate(Payload.from_input(arguments))
                output = evaluation.to_llm()
            elif tool_name == "apply_ptx_patch" and workspace is not None:
                try:
                    output = json.dumps(workspace.apply_patch(arguments))
                    resulting_payload = workspace.candidate.model_dump(
                        exclude_none=False
                    )
                except ValueError as error:
                    output = json.dumps({"error": str(error)})
            elif tool_name == "verify_current_ptx" and workspace is not None:
                evaluation = verifier.evaluate(
                    Payload.from_input(workspace.candidate.model_dump())
                )
                output = evaluation.to_llm()
            else:
                raise RuntimeError(f"Unsupported function call: {tool_name}")
            if cost_log_path is not None:
                record_tool_call(
                    Path(cost_log_path).parent,
                    provider="openai",
                    tool_name=tool_name,
                    call_id=_get_field(function_call, "call_id"),
                    payload=arguments,
                    answer=output,
                    resulting_payload=resulting_payload,
                )
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": _get_field(function_call, "call_id"),
                    "output": output,
                }
            )

        kwargs = {
            "model": model,
            "previous_response_id": _get_field(response, "id"),
            "input": tool_outputs,
            "tools": available_tools,
            "text": {"format": response_format},
        }
        if reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": reasoning_effort}
