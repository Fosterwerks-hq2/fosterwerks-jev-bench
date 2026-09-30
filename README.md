# Fast decisions, careful escalation

Can a fast, typed classifier (TypeSafe's **Jev**) make most prospect-triage decisions at a fraction
of an LLM's cost, and hand only the uncertain ones to Claude? This measures it on real data: the
a real prospect triage of Detroit small businesses (290 entities, tiers A/B/C/D/X, 29 hand-checked verdicts).

## Results (test split, 203 items)

| | Accuracy | Contact precision | Contact recall | Cost / 1k items | Median latency |
|---|---|---|---|---|---|
| Rules baseline | 51.2% | 70.7% | 89.2% | $0 | ~0 ms |
| Jev (jev-1.13.0, prompt v2) | 69.5% | 81.1% | 92.3% | $0.04 | 216 ms |
| Jev, escalate to Claude below 0.70 | 75.9% | 82.2% | 92.3% | $0.86 | 253 ms |
| Jev, escalate to Claude below 0.80 | 79.3% | 84.7% | 93.8% | $1.14 | 278 ms |
| Claude Opus 5.5 (effort low) | 80.3% | 84.7% | 93.8% | $2.55 | 2,058 ms |

The 0.70 threshold was chosen on the dev split; 0.80 is shown for reference. At 0.70 the cascade
sends 32% of items to Claude and keeps 94% of Claude's accuracy at a third of the cost.
"Contact" means tier A or B, the decision that actually triggers outreach.

What moved Jev from 57.6% to 69.5%: v1 said "when the evidence is thin, choose B", and Jev read it
literally, sending thin one-person and defunct entities to B. v2 scopes that rule to A versus B
and asks for a concrete sign of paid staff or revenue before B. Tuned on dev only.

Caveats: 203 test items (about 20 hand-checked), so differences of a couple of points are noise.
Cost and latency gaps are large enough to hold. Whole study cost under $1 in API calls.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m bench.dataset --source DIR # builds data/dataset.jsonl from the private triage files
```

## Run

```bash
.venv/bin/python -m bench.run --provider baseline --split all          # free, offline, checks the harness
.venv/bin/python -m bench.run --provider claude --split dev --limit 20  # small paid sample first
.venv/bin/python -m bench.run --provider claude --split test           # Claude baseline
.venv/bin/python -m bench.run --provider jev --split dev --limit 20     # needs TYPESAFE_API_KEY; ~$0.00004/item
.venv/bin/python -m bench.run --provider jev --split test              # pinned to jev-1.13.0
.venv/bin/python -m bench.evaluate --split test --fast jev__jev-1.13.0 --strong claude__claude-opus-5-5__low
```

`--model` and `--effort` compare Claude settings (e.g. `--model claude-haiku-4-5`). Every run
records cost and latency per item; reruns skip finished items.

## Method

- **Task:** from the evidence gathered for an entity, predict its tier and whether it's worth contacting (A or B).
- **Labels:** the triage tiers after the calibration pass. Accuracy on the 29 hand-checked items is reported separately.
- **No leakage:** the triage rationale, confidence and notes are excluded from the inputs.
- **Split:** stratified by tier, 30% dev for prompt work, 70% test for the headline numbers. Seed 7.
- **Cascade:** simulated from saved runs, so every threshold is compared at no extra cost.

## Privacy

`data/` and raw `results/*.jsonl` are gitignored: they name real businesses. Publish only
`results/report.html` or aggregate tables.
