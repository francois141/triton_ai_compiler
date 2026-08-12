from anthropic import Anthropic
from openai import OpenAI

from skills import (
    load_anthropic_ncu_report,
    load_anthropic_ptx,
    load_ncu_report,
    load_ptx,
)

from .response import request_anthropic_json, request_openai_json
from .setup import build_anthropic_tools, build_openai_tools


class ProviderSession:
    def __init__(self, client, tools, autotune_metrics=None):
        self.client = client
        self.tools = tools
        self.autotune_metrics = autotune_metrics

    def request_json(
        self,
        *,
        model,
        prompt,
        response_format,
        reasoning_effort,
        kernel_name,
        cost_log_path=None,
        pipeline=None,
        current_candidate=None,
    ):
        raise NotImplementedError


class OpenAIProviderSession(ProviderSession):
    def request_json(
        self,
        *,
        model,
        prompt,
        response_format,
        reasoning_effort,
        kernel_name,
        cost_log_path=None,
        pipeline=None,
        current_candidate=None,
    ):
        return request_openai_json(
            self.client,
            model=model,
            prompt=prompt,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            tools=self.tools,
            kernel_name=kernel_name,
            cost_log_path=cost_log_path,
            pipeline=pipeline,
            current_candidate=current_candidate,
            autotune_metrics=self.autotune_metrics,
        )


class AnthropicProviderSession(ProviderSession):
    def __init__(self, client, tools, skill_ids, autotune_metrics=None):
        super().__init__(client, tools, autotune_metrics)
        self.skill_ids = skill_ids

    def request_json(
        self,
        *,
        model,
        prompt,
        response_format,
        reasoning_effort,
        kernel_name,
        cost_log_path=None,
        pipeline=None,
        current_candidate=None,
    ):
        del reasoning_effort
        return request_anthropic_json(
            self.client,
            model=model,
            prompt=prompt,
            response_format=response_format,
            tools=self.tools,
            kernel_name=kernel_name,
            skill_ids=self.skill_ids,
            cost_log_path=cost_log_path,
            pipeline=pipeline,
            current_candidate=current_candidate,
            autotune_metrics=self.autotune_metrics,
        )


def create_provider_session(provider, *, autotune_metrics=None):
    if provider == "openai":
        client = OpenAI()
        skill_ids = [load_ptx(client), load_ncu_report(client)]
        print(f"=== Uploaded skills {', '.join(skill_ids)} ===", flush=True)
        return OpenAIProviderSession(
            client,
            build_openai_tools(skill_ids),
            autotune_metrics=autotune_metrics,
        )
    elif provider == "anthropic":
        client = Anthropic()
        skill_ids = [load_anthropic_ptx(client), load_anthropic_ncu_report(client)]
        print(f"=== Uploaded skills {', '.join(skill_ids)} ===", flush=True)
        return AnthropicProviderSession(
            client,
            build_anthropic_tools(),
            skill_ids,
            autotune_metrics=autotune_metrics,
        )

    raise ValueError(f"Unsupported provider: {provider}")
