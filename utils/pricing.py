from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


PriceTable = Mapping[str, Mapping[str, float]]

_PRICING_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    # Price estimates in USD per 1M tokens.
    # Keep these aligned with provider pricing pages.
    "gpt-5.6-sol": {
        "input": 5.00,
        "cached_input": 0.50,
        "cache_write": 6.25,
        "output": 30.00,
    },
    "gpt-5.6-terra": {
        "input": 2.50,
        "cached_input": 0.25,
        "cache_write": 3.125,
        "output": 15.00,
    },
    "gpt-5.6-luna": {
        "input": 1.00,
        "cached_input": 0.10,
        "cache_write": 1.25,
        "output": 6.00,
    },
    "gpt-5.5": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
    "gpt-5.3-codex": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.20, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.04, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.01, "output": 0.40},
    "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
    "o3": {"input": 10.00, "cached_input": 2.50, "output": 40.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "claude-fable-5": {"input": 10.00, "output": 50.00},
    "qwen/qwen3-coder": {"input": 0.22, "output": 1.80},
    "deepseek/deepseek-v4-pro": {
        "input": 0.435,
        "cached_input": 0.003625,
        "output": 0.87,
    },
}

PRICING_PER_1M_TOKENS: PriceTable = MappingProxyType(
    {
        model: MappingProxyType(pricing)
        for model, pricing in _PRICING_PER_1M_TOKENS.items()
    }
)


@dataclass(frozen=True, slots=True)
class TokenCounts:
    """Token usage values used for provider cost estimates.

    Args:
        input_tokens: Total prompt or input tokens.
        output_tokens: Total completion or output tokens.
        cached_input_tokens: Input tokens served from cache.
        cache_write_tokens: Input tokens written to cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0


def get_pricing(model: str) -> Mapping[str, float] | None:
    """Return the pricing row for a model.

    Args:
        model: Provider model identifier.

    Returns:
        Pricing in USD per 1M tokens, or None when the model is unknown.
    """

    return PRICING_PER_1M_TOKENS.get(model)


def get_price(model: str, token_type: str) -> float | None:
    """Return a single token price for a model.

    Args:
        model: Provider model identifier.
        token_type: Token price type, such as input, cached_input, or output.

    Returns:
        Price in USD per 1M tokens, or None when absent.
    """

    pricing = get_pricing(model)
    if pricing is None:
        return None
    return pricing.get(token_type)


def estimate_token_cost(model: str, token_counts: TokenCounts) -> float | None:
    """Estimate token cost for a provider response.

    Args:
        model: Provider model identifier.
        token_counts: Input, cached input, cache write, and output usage.

    Returns:
        Estimated USD cost, or None when pricing is unavailable.
    """

    pricing = get_pricing(model)
    if pricing is None:
        return None

    uncached_input_tokens = max(
        token_counts.input_tokens
        - token_counts.cached_input_tokens
        - token_counts.cache_write_tokens,
        0,
    )
    cached_input_price = pricing.get("cached_input", pricing["input"])
    cache_write_price = pricing.get("cache_write", 0.0)

    return (
        uncached_input_tokens * pricing["input"]
        + token_counts.cached_input_tokens * cached_input_price
        + token_counts.cache_write_tokens * cache_write_price
        + token_counts.output_tokens * pricing["output"]
    ) / 1_000_000
