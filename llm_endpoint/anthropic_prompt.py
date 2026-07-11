from anthropic import Anthropic

from utils.pricing import TokenCounts, estimate_token_cost

from .base import LLMEndpoint, PtxKernel


class AnthropicPrompt(LLMEndpoint):
    DEFAULT_MODEL = "claude-opus-4-8"

    def __init__(self, model=None, max_tokens=8192):
        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.client = Anthropic()

    def _estimate_cost(self, response):
        usage = getattr(response, "usage", None)

        if usage is None:
            return None

        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0

        return estimate_token_cost(
            self.model,
            TokenCounts(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    def generate_response(self, prompt, *, num_answers=None):
        requested = int(num_answers) if num_answers is not None else 1

        if requested <= 0:
            raise ValueError("num_answers must be positive when provided.")

        prompt = f"{prompt}\n\nGenerate exactly one answer dictionary as `ptx_kernel`."

        all_answers = []
        total_cost = 0.0
        cost_available = True

        for _ in range(requested):
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                output_format=PtxKernel,
            )

            cost = self._estimate_cost(response)
            if cost is None:
                cost_available = False
                print(f"Estimated query cost: unavailable for model {self.model!r}")
            else:
                total_cost += cost
                print(f"Estimated query cost: ${cost:.6f}")

            parsed = response.parsed_output
            if parsed is None:
                stop_reason = getattr(response, "stop_reason", None) or "unknown"
                raise ValueError(
                    "Anthropic did not return a PTX kernel "
                    f"(stop reason: {stop_reason})."
                )
            all_answers.append(parsed.model_dump(exclude_none=True))

        if cost_available:
            print(f"Estimated total query cost: ${total_cost:.6f}")
        else:
            print(f"Estimated total query cost: unavailable for model {self.model!r}")

        return all_answers
