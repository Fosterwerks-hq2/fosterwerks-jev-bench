"""Jev (TypeSafe AI) classifier: one System One request per entity, typed answers back.

Written against the TypeSafe docs as of 2026-09-29 (jev-1.13, typesafe-sdk 0.7). Jev reads its
questions literally and does badly with long, loosely relevant context, so it does not get the
whole rubric like Claude does. Each tier's test goes into its option description instead, and each
gate becomes a separate yes/no question.

One request per item asks three questions over the same evidence (they run in parallel, and cost is
per input token):
  tier   Choice over A/B/C/D/X. This is the prediction the benchmark scores.
  gate1  Noul: is the manual work paid for (a cost the client could name)?
  gate2  Noul: can the org fund a $2,250+ build from revenue or a program budget?
The gate probabilities go into `extra`, so a gates-then-tier composite can be tried offline later
without paying for another run.

Confidence: the probability Jev puts on the chosen tier, since the evaluator reads confidence as
"probability this tier is correct" (calibration error, cascade threshold). Jev's own `confidence`
number (how concentrated the probability is across options) is saved in `extra` as well.

Credentials: TYPESAFE_API_KEY (console.typesafe.ai).
"""
import os
import time

from .base import Prediction

DEFAULT_MODEL = "jev-1.13.0"  # pinned, not jev-latest, so the numbers stay tied to one version
PRICE_PER_MTOK_INPUT = {"jev-1.13.0": 0.042}  # output tokens are free (docs.typesafe.ai/models)

GATES = (
    "Fosterwerks sells fixed-price automation builds of $2,250 to $16,000. A prospect qualifies "
    "only if (1) someone is paid to do the manual work, or its volume has a cost the client could "
    "name in dollars, and (2) the build can be paid for out of recurring revenue or a recurring "
    "grant or program budget. An owner doing the work themselves for free is not a cost. Savings, "
    "a founder's credit card, or hoping the build pays for itself are not funding. Judge revenue "
    "and budget, not headcount or tax status."
)

TIER = {
    "instructions": {
        "task": "Which triage tier fits this entity, based only on the evidence given?",
        "qualification": GATES,
        "rule": "When the evidence is missing or thin, choose B, not A. Sounding techy is not evidence of budget.",
    },
    "criteria": {
        "A": "Contact now: evidence shows both conditions are met (paid staff, named revenue or budget, "
             "or funding) and there is visible manual or operational work worth automating.",
        "B": "Contact after checking: probably meets both conditions, but the evidence is thin and "
             "needs one more check before outreach.",
        "C": "Real, operating business that cannot plausibly fund a $2,250 build today.",
        "D": "Solo or side business: one person, no paid staff, and no manual-labor cost anyone pays for.",
        "X": "Not a prospect: defunct, cannot be found, only a person's name, a volunteer group with "
             "no paid staff, or an organization with no operations to automate.",
    },
}

GATE1 = (
    "Does the evidence show that someone is paid to do this entity's manual or operational work, "
    "or that the work's volume has a cost the entity could name in dollars? An owner doing it "
    "themselves for free does not count."
)
GATE2 = (
    "Does the evidence show that this entity has recurring revenue or a recurring grant or program "
    "budget that could pay for a $2,250 to $16,000 software build? Savings, personal credit, or "
    "early-stage hopes do not count."
)


class Jev:
    def __init__(self, model: str = DEFAULT_MODEL):
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise SystemExit("Set TYPESAFE_API_KEY (from console.typesafe.ai) to run Jev.")
        if model not in PRICE_PER_MTOK_INPUT:
            raise ValueError(f"No price on file for {model}; add it to PRICE_PER_MTOK_INPUT")
        from typesafe_sdk import Choice, Noul, TypeSafeClient

        self.model = model
        self.name = f"jev:{model}"
        self.client = TypeSafeClient(model=model)  # retries 429s with backoff by default
        self.questions = {
            "tier": Choice(instructions=TIER["instructions"], criteria=TIER["criteria"]),
            "gate1": Noul(instructions=GATE1),
            "gate2": Noul(instructions=GATE2),
        }

    def predict(self, item: dict) -> Prediction:
        import typesafe_sdk as ts

        # Named fields rather than one text blob; empty fields say so explicitly.
        state = {k: (v or "NONE") for k, v in item["input"].items()}
        t0 = time.perf_counter()
        try:
            resp = self.client.system_one(state, self.questions)
        except ts.TypeSafeRateLimitError as e:
            return Prediction(item["id"], self.name, None, None, 0.0, 0.0, error=f"rate_limited: {e}")
        except ts.TypeSafeAPIError as e:
            return Prediction(item["id"], self.name, None, None, 0.0, 0.0, error=f"api: {e}")
        except ts.TypeSafeError as e:
            return Prediction(item["id"], self.name, None, None, 0.0, 0.0, error=f"{type(e).__name__}: {e}")
        latency = (time.perf_counter() - t0) * 1000

        tier = resp.choices["tier"]
        cost = resp.usage.input_tokens * PRICE_PER_MTOK_INPUT[self.model] / 1_000_000
        extra = {
            "probabilities": tier.probabilities,
            "jev_confidence": tier.confidence,
            "gate1": resp.nouls["gate1"].noul,
            "gate2": resp.nouls["gate2"].noul,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "request_id": _request_id(resp),
            "served_by": resp.model,
        }
        return Prediction(item["id"], self.name, tier.choice, float(tier.probabilities[tier.choice]),
                          cost, latency, extra=extra)


def _request_id(resp):
    try:
        return resp.request_id  # read from a response header; the SDK raises if it's missing
    except Exception:
        return None
