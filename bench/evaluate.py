"""Score every saved run, then simulate the cascade: a fast model answers, and anything below a
confidence threshold escalates to a stronger model. The cascade is computed from saved results,
so trying thresholds costs nothing.

  python -m bench.evaluate --split test
  python -m bench.evaluate --split test --fast jev --strong claude__claude-opus-5-5__low
Writes results/report.html and prints the same tables.
"""
import argparse
import html
import json
import statistics
from pathlib import Path

from .dataset import TIERS, load
from .run import RESULTS

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def load_run(name: str) -> dict[str, dict]:
    preds = {}
    for line in open(RESULTS / f"{name}.jsonl", encoding="utf-8"):
        p = json.loads(line)
        preds[p["id"]] = p  # last write wins
    return preds


def metrics(items: list[dict], preds: dict[str, dict]) -> dict:
    scored = [(it, preds.get(it["id"])) for it in items]
    answered = [(it, p) for it, p in scored if p and p.get("tier")]
    n = len(items)
    if not answered:
        return {"n": n, "answered": 0}
    correct = sum(p["tier"] == it["label"] for it, p in answered)
    # Macro F1 over the five tiers
    f1s = []
    for t in TIERS:
        tp = sum(p["tier"] == t and it["label"] == t for it, p in answered)
        fp = sum(p["tier"] == t and it["label"] != t for it, p in answered)
        fn = sum(p["tier"] != t and it["label"] == t for it, p in answered)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    # "Worth contacting" (A or B) as a yes/no decision
    tp = sum(p["tier"] in "AB" and it["contact"] for it, p in answered)
    fp = sum(p["tier"] in "AB" and not it["contact"] for it, p in answered)
    fn = sum(p["tier"] not in "AB" and it["contact"] for it, p in answered)
    calib = [(it, p) for it, p in answered if it["calibrated"]]
    # Expected calibration error on stated confidence
    conf = [(p["confidence"], p["tier"] == it["label"]) for it, p in answered if p.get("confidence") is not None]
    ece = 0.0
    for lo in [i / 10 for i in range(10)]:
        bucket = [(c, ok) for c, ok in conf if lo <= c < lo + 0.1 or (lo == 0.9 and c == 1.0)]
        if bucket:
            ece += len(bucket) / len(conf) * abs(statistics.mean(c for c, _ in bucket) - statistics.mean(ok for _, ok in bucket))
    lat = sorted(p["latency_ms"] for _, p in answered)
    cost = sum(p["cost_usd"] for _, p in scored if p)
    return {
        "n": n,
        "answered": len(answered),
        "accuracy": correct / len(answered),
        "macro_f1": statistics.mean(f1s),
        "contact_precision": tp / (tp + fp) if tp + fp else 0.0,
        "contact_recall": tp / (tp + fn) if tp + fn else 0.0,
        "calibrated_accuracy": (sum(p["tier"] == it["label"] for it, p in calib) / len(calib)) if calib else None,
        "ece": ece if conf else None,
        "cost_per_1k": cost / max(1, len([p for _, p in scored if p])) * 1000,
        "p50_ms": lat[len(lat) // 2],
        "p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
    }


def cascade(items, fast, strong, threshold):
    """Fast model answers; low-confidence or failed items go to the strong model."""
    merged, escalated = {}, 0
    for it in items:
        f, s = fast.get(it["id"]), strong.get(it["id"])
        use_fast = f and f.get("tier") and (f.get("confidence") or 0) >= threshold
        if use_fast or not s:
            merged[it["id"]] = f
        else:
            escalated += 1
            s = dict(s)
            s["cost_usd"] += f["cost_usd"] if f else 0.0  # the fast call was paid for either way
            s["latency_ms"] += f["latency_ms"] if f else 0.0
            merged[it["id"]] = s
    m = metrics(items, {k: v for k, v in merged.items() if v})
    m["escalated_share"] = escalated / len(items)
    return m


def fmt(v, kind):
    if v is None:
        return "-"
    return {"pct": f"{v:.1%}", "usd": f"${v:.4f}", "ms": f"{v:,.0f}", "num": f"{v:.3f}"}[kind]


COLUMNS = [("accuracy", "Accuracy", "pct"), ("macro_f1", "Macro F1", "num"), ("contact_precision", "Contact precision", "pct"),
           ("contact_recall", "Contact recall", "pct"), ("calibrated_accuracy", "Hand-checked accuracy", "pct"),
           ("ece", "Calibration error", "num"), ("cost_per_1k", "Cost / 1k items", "usd"),
           ("p50_ms", "p50 ms", "ms"), ("p95_ms", "p95 ms", "ms")]


def table(rows: list[tuple[str, dict]], extra=None) -> tuple[str, str]:
    cols = COLUMNS + (extra or [])
    head = ["Run", "n"] + [c[1] for c in cols]
    body = [[name, f"{m.get('answered', 0)}/{m['n']}"] + [fmt(m.get(k), kind) for k, _, kind in cols] for name, m in rows]
    widths = [max(len(str(r[i])) for r in [head] + body) for i in range(len(head))]
    text = "\n".join("  ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in [head] + body)
    h = "<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in head) + "</tr></thead><tbody>"
    h += "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in body) + "</tbody></table>"
    return text, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--fast", help="Run name for the fast first pass (e.g. jev)")
    ap.add_argument("--strong", help="Run name for escalations (e.g. claude__claude-opus-5-5__low)")
    args = ap.parse_args()
    items = load(args.split)
    runs = sorted(p.stem for p in RESULTS.glob("*.jsonl"))
    if not runs:
        raise SystemExit("No runs in results/ yet. Start with: python -m bench.run --provider baseline --split all")

    rows = [(r, metrics(items, load_run(r))) for r in runs]
    text, h_runs = table(rows)
    print(f"\n== Runs on the {args.split} split ({len(items)} items)\n{text}")

    h_cascade = ""
    if args.fast and args.strong:
        fast, strong = load_run(args.fast), load_run(args.strong)
        crows = [(f"{args.fast} -> {args.strong} @ {t:.2f}", cascade(items, fast, strong, t)) for t in THRESHOLDS]
        text, h_cascade = table(crows, extra=[("escalated_share", "Escalated", "pct")])
        print(f"\n== Cascade: {args.fast} first, below the threshold goes to {args.strong}\n{text}")

    report = RESULTS / "report.html"
    report.write_text(f"""<!doctype html><meta charset="utf-8"><title>Fast Decisions Benchmark</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;margin:32px;color:#2C2F2D;background:#F3F4EF}}
table{{border-collapse:collapse;margin:12px 0 28px}}th,td{{padding:6px 10px;border-bottom:1px solid #CDD1CA;text-align:right}}
th:first-child,td:first-child{{text-align:left}}h1,h2{{font-weight:700}}</style>
<h1>Fast decisions, careful escalation</h1>
<p>Detroit small-business prospect triage, {args.split} split, {len(items)} items. Labels are the calibrated triage tiers.</p>
<h2>Runs</h2>{h_runs}{'<h2>Cascade</h2>' + h_cascade if h_cascade else ''}""", encoding="utf-8")
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
