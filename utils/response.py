import json
import time
from functools import cache

from openai import NotFoundError
from triton_ptx import Payload, TritonPTXCandidateEvaluator, resolve_kernel

from .cost import append_cost_log
from prompts.blocks import system_prompt


RESPONSE_RETRY_ATTEMPTS = 6


@cache
def verifier_for_kernel(kernel_name):
    return TritonPTXCandidateEvaluator(resolve_kernel(kernel_name))


def _get_field(value, field_name):
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


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


def request_json(
    client,
    *,
    model,
    prompt,
    response_format,
    reasoning_effort,
    tools,
    kernel_name,
):
    verifier = verifier_for_kernel(kernel_name)
    kwargs = {
        "model": model,
        "instructions": system_prompt(),
        "input": [{"role": "user", "content": prompt}],
        "tools": tools,
        "text": {"format": response_format},
    }
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    total_cost = None
    while True:
        for attempt in range(RESPONSE_RETRY_ATTEMPTS):
            try:
                response = client.responses.create(**kwargs)
                break
            except NotFoundError:
                if attempt == RESPONSE_RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(2**attempt)

        print_tool_calls(response)
        cost = append_cost_log(model=model, response=response)
        if cost is not None:
            total_cost = (total_cost or 0) + cost

        function_calls = [
            item
            for item in (_get_field(response, "output") or [])
            if _get_field(item, "type") == "function_call"
        ]
        if not function_calls:
            return response, total_cost

        tool_outputs = []
        for function_call in function_calls:
            if _get_field(function_call, "name") != "launch_verifier":
                raise RuntimeError(
                    f"Unsupported function call: {_get_field(function_call, 'name')}"
                )
            arguments = json.loads(_get_field(function_call, "arguments"))
            evaluation = verifier.evaluate(Payload.from_input(arguments))
            if hasattr(evaluation, "to_json"):
                evaluation = evaluation.to_json(indent=2)
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": _get_field(function_call, "call_id"),
                    "output": evaluation,
                }
            )

        kwargs = {
            "model": model,
            "previous_response_id": _get_field(response, "id"),
            "input": tool_outputs,
            "tools": tools,
            "text": {"format": response_format},
        }
        if reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": reasoning_effort}
