"""Build the benchmark dataset from a private prospect triage.

Each item is one entity: the evidence the research agents gathered (inputs) and the tier
they settled on after calibration (label). Fields that give the answer away (fw_rationale,
confidence, notes) are left out of the inputs.

Usage: python -m bench.dataset --source DIR
Writes data/dataset.jsonl and data/rubric.md (both gitignored).
"""
import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TIERS = ["A", "B", "C", "D", "X"]
INPUT_FIELDS = ["entity_name", "co_listed", "what_they_do", "sector", "website", "size_signal", "automation_surface"]


def build(source: Path, test_share: float = 0.7, seed: int = 7) -> list[dict]:
    rows = list(csv.DictReader(open(source / "data" / "02-triage-merged.csv", encoding="utf-8")))
    calibration = json.load(open(source / "data" / "05-calibration.json", encoding="utf-8"))
    calibrated = {(str(c["row_id"]), c["entity_name"]) for c in calibration}

    items = []
    for i, r in enumerate(rows):
        tier = r["fw_tier"].strip().upper()
        if tier not in TIERS:
            continue
        items.append({
            "id": f"tt-{i:03d}",
            "input": {k: r.get(k, "").strip() for k in INPUT_FIELDS},
            "label": tier,
            "contact": tier in ("A", "B"),
            "calibrated": (r["row_id"], r["entity_name"]) in calibrated,
            "source_confidence": r["confidence"],
        })

    # Stratified split so every tier appears in both dev and test.
    rng = random.Random(seed)
    by_tier = defaultdict(list)
    for it in items:
        by_tier[it["label"]].append(it)
    for group in by_tier.values():
        rng.shuffle(group)
        n_test = round(len(group) * test_share)
        for j, it in enumerate(group):
            it["split"] = "test" if j < n_test else "dev"
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Directory holding the private triage files")
    args = ap.parse_args()
    source = Path(args.source).expanduser()
    DATA.mkdir(exist_ok=True)
    items = build(source)
    with open(DATA / "dataset.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    shutil.copy(source / "RUBRIC.md", DATA / "rubric.md")
    counts = defaultdict(lambda: defaultdict(int))
    for it in items:
        counts[it["split"]][it["label"]] += 1
    print(f"{len(items)} items -> {DATA / 'dataset.jsonl'}")
    for split in ("dev", "test"):
        print(f"  {split}: " + ", ".join(f"{t}={counts[split][t]}" for t in TIERS))
    print(f"  calibrated (hand-checked) items: {sum(it['calibrated'] for it in items)}")


def load(split: str | None = None) -> list[dict]:
    items = [json.loads(line) for line in open(DATA / "dataset.jsonl", encoding="utf-8")]
    return [it for it in items if split in (None, "all", it["split"])]


if __name__ == "__main__":
    main()
