"""Jev (TypeSafe AI) classifier. PLACEHOLDER until early access comes through.

Jev returns typed values with a probability and a confidence score instead of text, which is
exactly what this benchmark compares. When access arrives:
  1. Read TypeSafe's API docs and install their client (add it to requirements.txt).
  2. Fill in `predict` below: send the rubric summary + format_item(item), ask for an enum
     output over ["A", "B", "C", "D", "X"], and map their probability to Prediction.confidence.
  3. Fill in PRICE from their pricing page, so cost per 1,000 items is comparable.
Nothing else in the harness needs to change.
"""
import os
import time

from .base import Prediction, format_item  # noqa: F401  (format_item is what you send)

PRICE_PER_CALL_USD = None  # TODO: from TypeSafe pricing


class Jev:
    name = "jev"

    def __init__(self):
        self.api_key = os.environ.get("JEV_API_KEY")
        if not self.api_key:
            raise SystemExit("Set JEV_API_KEY once TypeSafe early access is granted.")

    def predict(self, item: dict) -> Prediction:
        t0 = time.perf_counter()
        raise NotImplementedError(
            "Jev client not written yet: see the steps at the top of bench/providers/jev.py"
        )
        return Prediction(item["id"], self.name, None, None, 0.0, (time.perf_counter() - t0) * 1000)
