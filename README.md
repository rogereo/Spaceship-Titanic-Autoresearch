# Spaceship Titanic AutoResearch

A minimal Karpathy-style autoresearch loop that lets Claude Haiku 4.5 iteratively
improve a sklearn training script for the
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
Kaggle problem. Every iteration is logged as JSONL; an interactive HTML viewer
lets a human inspect the trace afterwards.

## Architectural contract

The system has three parts and they stay decoupled:

1. **The harness** ([`autoresearch_loop.py`](./autoresearch_loop.py)) — orchestrates the loop. Reads files, calls the Anthropic API, executes code, ratchets on success, reverts on failure, writes the trace.
2. **The problem definition** — [`prepare.py`](./prepare.py) (immutable evaluator), [`workspace/train.py`](./workspace/train.py) (the agent's sandbox, rewritten each turn), and [`program.md`](./program.md) (the research brief).
3. **The view** ([`index.html`](./index.html)) — single-file Plotly chart with click-to-expand iteration details, reads the JSONL trace.

The harness reads JSONL. The viewer reads JSONL. No shared imports.

## Layout

```
spaceship-titanic-autoresearch/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── prepare.py                  # immutable evaluator
├── program.md                  # research brief
├── autoresearch_loop.py        # the harness
├── index.html                  # the trace viewer
├── data/                       # gitignored; user places train.csv here
├── workspace/
│   ├── train.py                # agent's sandbox (starts as baseline)
│   └── notes.md                # gitignored; agent appends one line per turn
└── traces/                     # gitignored; one JSONL per run
```

## Setup

1. Clone the repo and `cd` into it.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and put your Anthropic API key in it. Set a hard
   spending cap on the key in the Anthropic console (recommended: $10).
5. Download `train.csv` from the
   [Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic/data)
   and place it at `data/train.csv`.
6. Make sure your working tree is clean and committed before running the harness
   — it creates a new branch and commits per iteration.

## Run

```powershell
python workspace\train.py                          # smoke test the baseline (~0.78)
python autoresearch_loop.py --max-iterations 1     # smoke test the full loop
python autoresearch_loop.py                        # real run (15 iters, capped at $10)
```

Each run creates a fresh branch `autoresearch/run-<timestamp>` and writes one
JSONL line per iteration to `traces/run_<timestamp>.jsonl`. Reverted iterations
are hard-reset out of git history; the kept iterations form a clean staircase.

## View the trace

```powershell
python -m http.server --bind 127.0.0.1     # serve the repo so fetch() can read the JSONL
```

Then open `http://localhost:8000/`.

The viewer auto-discovers trace files in `traces/`. With one trace it loads it
directly; with several it shows a picker. To deep-link a specific trace use
`http://localhost:8000/?trace=traces/run_<timestamp>.jsonl`.

> The `--bind 127.0.0.1` flag matters: without it, `http.server` advertises
> `http://[::]:8000/`, which Chrome refuses with `ERR_ADDRESS_INVALID`.

The chart shows attempt scores as colored dots (green=kept, grey=reverted,
red=crashed, orange=parse_failed) and the running best as a staircase line.
Click any dot to see that iteration's reflection, observations, hypothesis,
plan, code, and run output.

## Hard rules (enforced by the harness)

- `train.py` must call `prepare.evaluate(predict_fn)` exactly once and print
  `VAL_ACCURACY: 0.XXXX` as the last line of stdout.
- 60-second per-iteration timeout; longer runs are killed and reverted.
- Score regressions are reverted via `git reset --hard HEAD~1`.
- $10 total cost cap per run.
- Stops after 5 iterations with no improvement.

## Stop conditions

The loop ends when any of these is hit:
- `--max-iterations` reached (default 15)
- 5 consecutive iterations without improvement
- $10 total cost cap

## Blog post

This codebase is a companion to a forthcoming blog post on autoresearch. (Link
to be added.)
