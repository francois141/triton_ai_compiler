from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


PriceTable = Mapping[str, Mapping[str, float]]

_PRICING_PER_1M_TOKENS = {
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

PRICING_PER_1M_TOKENS = MappingProxyType(
    {
        model: MappingProxyType(pricing)
        for model, pricing in _PRICING_PER_1M_TOKENS.items()
    }
)


@dataclass(frozen=True, slots=True)
class TokenCounts:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0


def get_pricing(model):
    return PRICING_PER_1M_TOKENS.get(model)


def get_price(model, token_type):
    pricing = get_pricing(model)
    if pricing is None:
        return None
    return pricing.get(token_type)


def estimate_token_cost(model, token_counts):
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


COST_LOG_PATH = Path(__file__).resolve().parent.parent / "costs.txt"
WEB_SEARCH_COST_PER_CALL = 10.00 / 1_000


def _get_nested_int(value, *keys):
    for key in keys:
        if value is None:
            return 0
        if isinstance(value, dict):
            value = value.get(key)
        else:
            value = getattr(value, key, None)
    return int(value or 0)


def estimate_response_cost(response, model):
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "output_tokens": 0,
            "web_search_calls": 0,
        }

    input_tokens = _get_nested_int(usage, "input_tokens") or _get_nested_int(
        usage, "prompt_tokens"
    )
    output_tokens = _get_nested_int(usage, "output_tokens") or _get_nested_int(
        usage, "completion_tokens"
    )
    cached_input_tokens = _get_nested_int(
        usage, "input_tokens_details", "cached_tokens"
    ) or _get_nested_int(
        usage, "prompt_tokens_details", "cached_tokens"
    ) or _get_nested_int(usage, "cache_read_input_tokens")
    cache_write_tokens = _get_nested_int(
        usage, "input_tokens_details", "cache_write_tokens"
    ) or _get_nested_int(
        usage, "prompt_tokens_details", "cache_write_tokens"
    ) or _get_nested_int(usage, "cache_creation_input_tokens")
    tool_usage = getattr(response, "tool_usage", None)
    web_search_calls = _get_nested_int(
        tool_usage, "web_search", "num_requests"
    ) or _get_nested_int(tool_usage, "web_search", "requests")
    token_counts = {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_write_tokens": cache_write_tokens,
        "output_tokens": output_tokens,
        "web_search_calls": web_search_calls,
    }
    token_cost = estimate_token_cost(
        model,
        TokenCounts(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_write_tokens=cache_write_tokens,
        ),
    )
    if token_cost is None:
        return None, token_counts
    return token_cost + web_search_calls * WEB_SEARCH_COST_PER_CALL, token_counts


def read_daily_total(cost_log_path, date_text):
    if not cost_log_path.exists():
        return 0.0
    total = 0.0
    with cost_log_path.open(encoding="utf-8") as cost_log:
        for line in cost_log:
            if not line.startswith(f"{date_text}T") or " cost_usd=" not in line:
                continue
            cost_text = line.split(" cost_usd=", maxsplit=1)[1].split()[0]
            try:
                total += float(cost_text)
            except ValueError:
                continue
    return total


def append_cost_log(*, model, response, cost_log_path=COST_LOG_PATH):
    timestamp = datetime.now().astimezone()
    cost, token_counts = estimate_response_cost(response, model)
    daily_total = read_daily_total(cost_log_path, timestamp.date().isoformat())
    daily_total += cost or 0.0
    cost_log_path.parent.mkdir(parents=True, exist_ok=True)
    cost_text = "unavailable" if cost is None else f"{cost:.8f}"
    with cost_log_path.open("a", encoding="utf-8") as cost_log:
        cost_log.write(
            f"{timestamp.isoformat()} model={model} "
            f"input_tokens={token_counts['input_tokens']} "
            f"cached_input_tokens={token_counts['cached_input_tokens']} "
            f"cache_write_tokens={token_counts['cache_write_tokens']} "
            f"output_tokens={token_counts['output_tokens']} "
            f"web_search_calls={token_counts['web_search_calls']} "
            f"cost_usd={cost_text} daily_total_usd={daily_total:.8f}\n"
        )
    return cost


def append_daily_cost_summary(cost_log_path=COST_LOG_PATH, *, label="run_end"):
    timestamp = datetime.now().astimezone()
    daily_total = read_daily_total(cost_log_path, timestamp.date().isoformat())
    cost_log_path.parent.mkdir(parents=True, exist_ok=True)
    with cost_log_path.open("a", encoding="utf-8") as cost_log:
        cost_log.write(
            f"{timestamp.isoformat()} {label} daily_total_usd={daily_total:.8f}\n"
        )
