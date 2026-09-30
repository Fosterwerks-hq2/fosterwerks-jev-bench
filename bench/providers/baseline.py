"""A free, offline rule-of-thumb classifier. It exists to test the harness end to end
and to give a floor that any model has to beat."""
import re
import time

from .base import Prediction

NUMBER = re.compile(r"\d")
MONEY = re.compile(r"\$\s?\d|revenue|budget|grant|award|990|employees|staff", re.I)


class Baseline:
    name = "baseline-rules"

    def predict(self, item: dict) -> Prediction:
        t0 = time.perf_counter()
        inp = item["input"]
        what = inp["what_they_do"].upper()
        size = inp["size_signal"]
        surface = inp["automation_surface"].upper()
        if what == "UNKNOWN" or not what:
            tier, conf = "X", 0.6
        elif "NONE IDENTIFIED" in surface:
            tier, conf = "D", 0.5
        elif MONEY.search(size) and NUMBER.search(size):
            tier, conf = "B", 0.45
        else:
            tier, conf = "C", 0.35
        return Prediction(item["id"], self.name, tier, conf, 0.0, (time.perf_counter() - t0) * 1000)
