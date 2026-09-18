import os

from anthropic import Anthropic
from openai import OpenAI

from skills import (
    load_anthropic_ncu_report,
    load_anthropic_ptx,
    load_ncu_report,
    load_ptx,
)

from .response import (
    request_anthropic_json,
    request_openai_json,
    request_openrouter_json,
)
from .setup import build_anthropic_tools, build_openai_tools, build_openrouter_tools


class ProviderSession:
    def __init__(
        self,
        client,
        tools,
        autotune_metrics=None,
        *,
        enable_ncu_report=True,
        enable_sanitizer=True,
    ):
        self.client = client
        self.tools = tools
        self.autotune_metrics = autotune_metrics
        self.enable_ncu_report = enable_ncu_report
        self.enable_sanitizer = enable_sanitizer

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
        system_instruction=None,
        max_budget_usd=None,
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
        system_instruction=None,
        max_budget_usd=None,
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
            system_instruction=system_instruction,
            enable_ncu_report=self.enable_ncu_report,
            enable_sanitizer=self.enable_sanitizer,
            max_budget_usd=max_budget_usd,
        )


class AnthropicProviderSession(ProviderSession):
    def __init__(
        self,
        client,
        tools,
        skill_ids,
        autotune_metrics=None,
        *,
        enable_ncu_report=True,
        enable_sanitizer=True,
    ):
        super().__init__(
            client,
            tools,
            autotune_metrics,
            enable_ncu_report=enable_ncu_report,
            enable_sanitizer=enable_sanitizer,
        )
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
        system_instruction=None,
        max_budget_usd=None,
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
            system_instruction=system_instruction,
            enable_ncu_report=self.enable_ncu_report,
            enable_sanitizer=self.enable_sanitizer,
            max_budget_usd=max_budget_usd,
        )


class OpenRouterProviderSession(ProviderSession):
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
        system_instruction=None,
        max_budget_usd=None,
    ):
        del reasoning_effort
        return request_openrouter_json(
            self.client,
            model=model,
            prompt=prompt,
            response_format=response_format,
            tools=self.tools,
            kernel_name=kernel_name,
            cost_log_path=cost_log_path,
            pipeline=pipeline,
            current_candidate=current_candidate,
            autotune_metrics=self.autotune_metrics,
            system_instruction=system_instruction,
            enable_ncu_report=self.enable_ncu_report,
            enable_sanitizer=self.enable_sanitizer,
            max_budget_usd=max_budget_usd,
        )


def create_provider_session(
    provider,
    *,
    autotune_metrics=None,
    enable_ncu_skill=True,
    enable_ptx_skill=True,
    enable_ncu_report=True,
    enable_sanitizer=True,
):
    if provider == "openai":
        client = OpenAI()
        skill_ids = []
        if enable_ptx_skill:
            skill_ids.append(load_ptx(client))
        if enable_ncu_skill:
            skill_ids.append(load_ncu_report(client))
        print(f"=== Uploaded skills {', '.join(skill_ids)} ===", flush=True)
        return OpenAIProviderSession(
            client,
            build_openai_tools(skill_ids),
            autotune_metrics=autotune_metrics,
            enable_ncu_report=enable_ncu_report,
            enable_sanitizer=enable_sanitizer,
        )
    elif provider == "anthropic":
        client = Anthropic()
        skill_ids = []
        if enable_ptx_skill:
            skill_ids.append(load_anthropic_ptx(client))
        if enable_ncu_skill:
            skill_ids.append(load_anthropic_ncu_report(client))
        print(f"=== Uploaded skills {', '.join(skill_ids)} ===", flush=True)
        return AnthropicProviderSession(
            client,
            build_anthropic_tools(),
            skill_ids,
            autotune_metrics=autotune_metrics,
            enable_ncu_report=enable_ncu_report,
            enable_sanitizer=enable_sanitizer,
        )
    elif provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY must be set for --provider openrouter."
            )
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"X-OpenRouter-Title": "Triton PTX Agent"},
        )
        skill_names = [
            name
            for enabled, name in (
                (enable_ptx_skill, "ptx"),
                (enable_ncu_skill, "ncu"),
            )
            if enabled
        ]
        print(
            f"=== Enabled local skills {', '.join(skill_names) or 'none'} ===",
            flush=True,
        )
        return OpenRouterProviderSession(
            client,
            build_openrouter_tools(
                enable_ptx_skill=enable_ptx_skill,
                enable_ncu_skill=enable_ncu_skill,
            ),
            autotune_metrics=autotune_metrics,
            enable_ncu_report=enable_ncu_report,
            enable_sanitizer=enable_sanitizer,
        )

    raise ValueError(f"Unsupported provider: {provider}")
