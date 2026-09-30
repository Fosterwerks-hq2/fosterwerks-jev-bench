"""Run one provider over the dataset and save its predictions.

Results append to results/<run>.jsonl and a rerun skips items already done, so a run that
hits a rate limit can simply be restarted.

  python -m bench.run --provider baseline --split all
  python -m bench.run --provider claude --model claude-opus-5-5 --effort low --split dev --limit 20
  python -m bench.run --provider jev --split test
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .dataset import load

RESULTS = Path(__file__).resolve().parent.parent / "results"


def make_provider(args):
    if args.provider == "baseline":
        from .providers.baseline import Baseline
        return Baseline()
    if args.provider == "claude":
        from .providers.claude import Claude
        return Claude(model=args.model, effort=args.effort)
    if args.provider == "jev":
        from .providers.jev import Jev
        return Jev(model=args.model) if args.model.startswith("jev") else Jev()
    raise SystemExit(f"Unknown provider {args.provider}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["baseline", "claude", "jev"])
    ap.add_argument("--model", default="claude-opus-5-5", help="Claude model, or a Jev model ID with --provider jev")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--limit", type=int, default=0, help="Only the first N items (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    provider = make_provider(args)
    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / (provider.name.replace(":", "__") + ".jsonl")
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in open(out_path) if json.loads(l).get("tier")}

    items = [it for it in load(args.split) if it["id"] not in done]
    if args.limit:
        items = items[: args.limit]
    print(f"{provider.name}: {len(items)} to run ({len(done)} already done) -> {out_path.name}")

    cost = 0.0
    errors = 0
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(provider.predict, it) for it in items]
        for n, fut in enumerate(as_completed(futures), 1):
            p = fut.result()
            f.write(json.dumps(p.to_dict()) + "\n")
            f.flush()
            cost += p.cost_usd
            errors += p.error is not None
            if n % 25 == 0 or n == len(items):
                print(f"  {n}/{len(items)}  spent ${cost:.4f}  errors {errors}")


if __name__ == "__main__":
    main()
