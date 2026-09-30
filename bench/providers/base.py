"""What every provider returns, so the evaluator can compare them like for like."""
from dataclasses import asdict, dataclass, field


@dataclass
class Prediction:
    id: str
    provider: str
    tier: str | None          # A/B/C/D/X, or None if the call failed
    confidence: float | None  # 0..1. Jev: model probability. Claude: self-reported.
    cost_usd: float
    latency_ms: float
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def format_item(item: dict) -> str:
    """The evidence for one entity, as plain text. Same text for every provider."""
    inp = item["input"]
    lines = [f"{k.replace('_', ' ')}: {v or 'NONE'}" for k, v in inp.items()]
    return "\n".join(lines)
