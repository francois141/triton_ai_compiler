import os

from openai import OpenAI

from utils.pricing import TokenCounts, estimate_token_cost

from .base import LLMEndpoint, PtxKernel


class OpenRouterPrompt(LLMEndpoint):
    DEFAULT_MODEL = "qwen/qwen3-coder"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model=None,
        api_key=None,
        site_url=None,
        app_name=None,
    ):
        self.model = model or self.DEFAULT_MODEL

        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OpenRouter support requires OPENROUTER_API_KEY or an api_key option."
            )

        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-OpenRouter-Title"] = app_name

        self.client = OpenAI(
            api_key=key,
            base_url=self.BASE_URL,
            default_headers=default_headers or None,
        )

    def _estimate_cost(self, response):
        usage = getattr(response, "usage", None)

        if usage is None:
            return None

        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached_input_tokens = 0

        if prompt_details is not None:
            cached_input_tokens = getattr(prompt_details, "cached_tokens", 0) or 0

        return estimate_token_cost(
            self.model,
            TokenCounts(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
            ),
        )

    def generate_response(self, prompt, *, num_answers=None):
        requested = int(num_answers) if num_answers is not None else 1

        if requested <= 0:
            raise ValueError("num_answers must be positive when provided.")

        prompt = f"{prompt}\n\nGenerate exactly one answer dictionary as `ptx_kernel`."

        request_kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "n": requested,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ptx_kernel",
                    "strict": True,
                    "schema": PtxKernel.model_json_schema(),
                },
            },
        }

        response = self.client.chat.completions.create(**request_kwargs)

        cost = self._estimate_cost(response)
        if cost is None:
            print(f"Estimated query cost: unavailable for model {self.model!r}")
        else:
            print(f"Estimated total query cost: ${cost:.6f}")

        all_answers = []
        for choice in response.choices:
            text = choice.message.content or ""
            if not text:
                refusal = getattr(choice.message, "refusal", None)
                detail = refusal or "No structured output returned."
                raise ValueError(f"OpenRouter did not return a PTX kernel: {detail}")

            parsed = PtxKernel.model_validate_json(text)
            all_answers.append(parsed.model_dump(exclude_none=True))

        return all_answers
