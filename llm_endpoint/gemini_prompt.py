from utils.response_format import PTX_KERNEL_JSON_SCHEMA, PtxKernel

from .base import LLMEndpoint


class GeminiPrompt(LLMEndpoint):
    DEFAULT_MODEL = "gemini-2.5-pro"

    def __init__(self, model=None):
        self.model = model or self.DEFAULT_MODEL
        self._types = None
        self.client = self._build_client()

    def _build_client(self):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "Gemini support requires the `google-genai` package."
            ) from exc

        self._types = types
        return genai.Client()

    def _estimate_cost(self, response):
        # Pricing is model-dependent and changes independently from the SDK.
        # Return None unless the project chooses to maintain a pricing table.
        return None

    def _extract_text(self, response):
        text = getattr(response, "text", None)
        if text:
            return text

        parts = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(part_text)
        return "\n".join(parts)

    def generate_response(self, prompt, *, num_answers=None):
        requested = int(num_answers) if num_answers is not None else 1

        if requested <= 0:
            raise ValueError("num_answers must be positive when provided.")

        prompt = f"{prompt}\n\nGenerate exactly one answer dictionary as `ptx_kernel`."

        all_answers = []
        total_cost = 0.0
        cost_available = True

        config = self._types.GenerateContentConfig(
            candidate_count=1,
            response_mime_type="application/json",
            response_json_schema=PTX_KERNEL_JSON_SCHEMA,
        )

        for _ in range(requested):
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

            cost = self._estimate_cost(response)
            if cost is None:
                cost_available = False
                print(f"Estimated query cost: unavailable for model {self.model!r}")
            else:
                total_cost += cost
                print(f"Estimated query cost: ${cost:.6f}")

            text = self._extract_text(response)
            if not text:
                raise ValueError("Gemini did not return a PTX kernel.")

            parsed = PtxKernel.model_validate_json(text)
            all_answers.append(parsed.model_dump(exclude_none=True))

        if cost_available:
            print(f"Estimated total query cost: ${total_cost:.6f}")
        else:
            print(f"Estimated total query cost: unavailable for model {self.model!r}")

        return all_answers
