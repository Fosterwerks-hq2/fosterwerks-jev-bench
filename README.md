# Fast decisions, careful escalation

Can a fast, typed classifier (TypeSafe's **Jev**) make most prospect-triage decisions at a fraction
of an LLM's cost, and hand only the uncertain ones to Claude? This measures it on real data: the
a real prospect triage of Detroit small businesses (290 entities, tiers A/B/C/D/X, 29 hand-checked verdicts).

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
.venv/bin/python -m bench.run --provider jev --split test              # once JEV_API_KEY exists
.venv/bin/python -m bench.evaluate --split test --fast jev --strong claude__claude-opus-5-5__low
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
