"""Claude classifier: the rubric as a cached system prompt, a typed JSON answer back.

Model and effort are flags so the cost/accuracy trade-off can be measured, not assumed.
Credentials: ANTHROPIC_API_KEY or an `ant auth login` profile.
"""
import json
import time

import anthropic

from ..dataset import DATA
from .base import Prediction, format_item

# $ per million tokens: (input, output). Cache writes bill at 1.25x input, cache reads at 0.1x.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
# Models that take output_config.effort and the server-side refusal fallback.
EFFORT_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}

SCHEMA = {
    "type": "object",
    "properties": {
        "tier": {"type": "string", "enum": ["A", "B", "C", "D", "X"]},
        "confidence": {"type": "number", "description": "Probability 0 to 1 that the tier is correct"},
    },
    "required": ["tier", "confidence"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You score prospects for Fosterwerks using the rubric below. You get the evidence \
already gathered for one entity. Do not research further. Return the tier the rubric gives it and \
your probability (0 to 1) that the tier is correct. Score conservatively: thin evidence is B, not A.

"""


class Claude:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low"):
        if model not in PRICES:
            raise ValueError(f"No price on file for {model}; add it to PRICES")
        self.model, self.effort = model, effort
        self.name = f"claude:{model}:{effort}" if model in EFFORT_MODELS else f"claude:{model}"
        self.client = anthropic.Anthropic()
        rubric = (DATA / "rubric.md").read_text(encoding="utf-8")
        # Frozen system prompt so every call after the first reads it from cache.
        self.system = [{"type": "text", "text": INSTRUCTIONS + rubric, "cache_control": {"type": "ephemeral"}}]

    def _cost(self, usage) -> float:
        p_in, p_out = PRICES[self.model]
        cache_w = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_r = getattr(usage, "cache_read_input_tokens", 0) or 0
        return (usage.input_tokens * p_in + cache_w * p_in * 1.25 + cache_r * p_in * 0.1
                + usage.output_tokens * p_out) / 1_000_000

    def predict(self, item: dict) -> Prediction:
        kwargs = dict(
            model=self.model,
            max_tokens=2048,
            system=self.system,
            messages=[{"role": "user", "content": format_item(item)}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        if self.model in EFFORT_MODELS:
            kwargs["output_config"]["effort"] = self.effort
            # Server-side fallback: if a safety classifier declines, the API retries on a fallback model.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        t0 = time.perf_counter()
        try:
            resp = (self.client.beta.messages.create(**kwargs) if "betas" in kwargs
                    else self.client.messages.create(**kwargs))
        except anthropic.RateLimitError as e:
            return Prediction(item["id"], self.name, None, None, 0.0, 0.0, error=f"rate_limited: {e.message}")
        except anthropic.APIStatusError as e:
            return Prediction(item["id"], self.name, None, None, 0.0, 0.0, error=f"api_{e.status_code}: {e.message}")
        except anthropic.APIConnectionError as e:
            return Prediction(item["id"], self.name, None, None, 0.0, 0.0, error=f"connection: {e}")
        latency = (time.perf_counter() - t0) * 1000
        cost = self._cost(resp.usage)
        extra = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                 "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
                 "request_id": resp._request_id, "served_by": resp.model}
        if resp.stop_reason == "refusal":
            return Prediction(item["id"], self.name, None, None, cost, latency, error="refusal", extra=extra)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            return Prediction(item["id"], self.name, None, None, cost, latency, error="bad_json", extra=extra)
        return Prediction(item["id"], self.name, out["tier"], float(out["confidence"]), cost, latency, extra=extra)
