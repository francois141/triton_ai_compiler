from .response import request_anthropic_json, request_openai_json
from .setup import build_anthropic_tools, build_openai_tools
from openai import OpenAI
from anthropic import Anthropic
from skills import load_ptx, load_anthropic_ptx


class ProviderSession:
    def __init__(self, client, tools):
        self.client = client
        self.tools = tools

    def request_json(
        self,
        *,
        model,
        prompt,
        response_format,
        reasoning_effort,
        kernel_name,
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
            current_candidate=current_candidate,
        )


class AnthropicProviderSession(ProviderSession):
    def __init__(self, client, tools, skill_id):
        super().__init__(client, tools)
        self.skill_id = skill_id

    def request_json(
        self,
        *,
        model,
        prompt,
        response_format,
        reasoning_effort,
        kernel_name,
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
            skill_id=self.skill_id,
            current_candidate=current_candidate,
        )


def create_provider_session(provider):
    if provider == "openai":
        client = OpenAI()
        skill_id = load_ptx(client)
        print(f"=== Uploaded PTX skill {skill_id} ===", flush=True)
        return OpenAIProviderSession(client, build_openai_tools(skill_id))
    elif provider == "anthropic":
        client = Anthropic()
        skill_id = load_anthropic_ptx(client)
        print(f"=== Uploaded PTX skill {skill_id} ===", flush=True)
        return AnthropicProviderSession(
            client,
            build_anthropic_tools(),
            skill_id,
        )

    raise ValueError(f"Unsupported provider: {provider}")
