from time import perf_counter

from openai import OpenAI

from utils.pricing import TokenCounts, estimate_token_cost

from utils.response_format import PtxKernel

from .base import LLMEndpoint


class OpenAIPrompt(LLMEndpoint):
    DEFAULT_MODEL = "gpt-5.6-sol"
    ALLOWED_REASONING_EFFORTS = {None, "minimal", "low", "medium", "high"}

    def __init__(
        self,
        model=None,
        reasoning_effort="medium",
    ):
        self.model = model or self.DEFAULT_MODEL

        if reasoning_effort not in self.ALLOWED_REASONING_EFFORTS:
            allowed = sorted(v for v in self.ALLOWED_REASONING_EFFORTS if v is not None)
            raise ValueError(
                f"Invalid reasoning_effort. Expected one of {allowed} or None."
            )

        self.reasoning_effort = reasoning_effort

        self.client = OpenAI()

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
            cache_write_tokens = (
                getattr(prompt_details, "cache_write_tokens", 0) or 0
            )
        else:
            cache_write_tokens = 0

        return estimate_token_cost(
            self.model,
            TokenCounts(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                cache_write_tokens=cache_write_tokens,
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
            "response_format": PtxKernel,
        }

        if self.reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = self.reasoning_effort

        print("======== Sending prompt to OpenAI ========", flush=True)
        request_start = perf_counter()
        response = self.client.chat.completions.parse(**request_kwargs)
        request_duration = perf_counter() - request_start
        print(
            "======== Received response from OpenAI "
            f"in {request_duration:.3f}s ========",
            flush=True,
        )

        cost = self._estimate_cost(response)
        if cost is None:
            print(f"Estimated query cost: unavailable for model {self.model!r}")
        else:
            print(f"Estimated total query cost: ${cost:.6f}")

        all_answers = []
        for choice in response.choices:
            parsed = choice.message.parsed
            if parsed is None:
                refusal = choice.message.refusal or "No structured output returned."
                raise ValueError(f"OpenAI did not return a PTX kernel: {refusal}")
            all_answers.append(parsed.model_dump(exclude_none=True))

        return all_answers
