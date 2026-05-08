"""
autoresearch_loop.py — the harness.

Reads program.md + workspace/train.py + workspace/notes.md + recent trace,
calls Claude Haiku 4.5, parses the XML response, writes the new train.py,
commits, runs it with a timeout, ratchets on success / reverts on failure,
appends one JSONL record per iteration to traces/run_<timestamp>.jsonl.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import anthropic


# -------------------------- Config --------------------------

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
TRACES = ROOT / "traces"
PROGRAM_MD = ROOT / "program.md"
TRAIN_PY = WORKSPACE / "train.py"
NOTES_MD = WORKSPACE / "notes.md"
BEST_JSON = ROOT / "best.json"

MODEL = "claude-haiku-4-5-20251001"
MAX_ITERATIONS = 15
NO_IMPROVEMENT_LIMIT = 5
COST_CAP_USD = 10.0
TIMEOUT_SECONDS = 60
MAX_TOKENS = 3000
RECENT_ITERS_IN_PROMPT = 3

# Haiku 4.5 pricing — adjust if Anthropic publishes different rates.
HAIKU_INPUT_PRICE_PER_MTOK = 1.0
HAIKU_OUTPUT_PRICE_PER_MTOK = 5.0

STDOUT_TRUNC = 3000
STDERR_TRUNC = 1500
NOTES_TRUNC = 2000

SYSTEM_PROMPT = """You are an ML research agent. You will be given:
1. A research brief (program.md) describing the problem, rules, and directives.
2. The current state of train.py (the script you are improving).
3. A short summary of recent iterations (what was tried, what worked).
4. Any notes you have accumulated.

Your job is to propose ONE concrete improvement to train.py per turn, in a strict
output format. Follow program.md's rules absolutely."""


# -------------------------- Utilities --------------------------

def read_text(path, default=""):
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def truncate_tail(s, n):
    if s is None:
        return ""
    return s if len(s) <= n else s[-n:]


def run(cmd, **kwargs):
    """Run a subprocess, return CompletedProcess. Defaults: capture, text, no shell."""
    return subprocess.run(
        cmd, capture_output=True, text=True, **kwargs
    )


# -------------------------- Git helpers --------------------------

def git(*args, check=True):
    res = run(["git", *args], cwd=ROOT)
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{res.stderr}")
    return res


def ensure_clean_repo():
    res = git("status", "--porcelain")
    if res.stdout.strip():
        raise RuntimeError(
            "Working tree has uncommitted changes. Commit or stash before running:\n"
            + res.stdout
        )


def create_run_branch():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    branch = f"autoresearch/run-{timestamp}"
    git("checkout", "-b", branch)
    return branch


def git_commit_all(message):
    git("add", "-A")
    res = run(["git", "commit", "-m", message], cwd=ROOT)
    if res.returncode != 0:
        return None
    sha = git("rev-parse", "--short", "HEAD").stdout.strip()
    return sha


def git_revert_last():
    git("reset", "--hard", "HEAD~1")


# -------------------------- Response parsing --------------------------

TAGS = ["reflection", "observations", "hypothesis", "plan", "code", "notes_append"]


def parse_response(text):
    """Extract the six XML tags. Returns dict of strings (or None if missing).

    Raises ValueError if any required tag is missing.
    """
    out = {}
    for tag in TAGS:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if not m:
            raise ValueError(f"missing <{tag}> tag")
        out[tag] = m.group(1).strip()

    # Strip the python code fence inside <code> if present.
    code = out["code"]
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
    if fence:
        code = fence.group(1)
    out["code"] = code.strip() + "\n"

    # Observations as bullet list.
    obs_lines = [
        re.sub(r"^[-*•]\s*", "", ln).strip()
        for ln in out["observations"].splitlines()
        if ln.strip()
    ]
    out["observations_list"] = obs_lines

    return out


# -------------------------- API call --------------------------

def call_claude(client, user_prompt):
    """Call Claude with one retry on rate-limit. Returns (raw_text, usage)."""
    for attempt in range(2):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            usage = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
            return text, usage
        except anthropic.RateLimitError:
            if attempt == 0:
                time.sleep(10)
                continue
            raise


def cost_of(usage):
    if HAIKU_INPUT_PRICE_PER_MTOK is None or HAIKU_OUTPUT_PRICE_PER_MTOK is None:
        return None
    return (
        usage["input"] / 1_000_000 * HAIKU_INPUT_PRICE_PER_MTOK
        + usage["output"] / 1_000_000 * HAIKU_OUTPUT_PRICE_PER_MTOK
    )


# -------------------------- Train.py execution --------------------------

SCORE_RE = re.compile(r"VAL_ACCURACY:\s*([\d.]+)")


def execute_train():
    """Run python workspace/train.py with timeout. Returns dict."""
    t0 = time.time()
    try:
        res = subprocess.run(
            [sys.executable, "train.py"],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        stdout = res.stdout
        stderr = res.stderr
        returncode = res.returncode
        timed_out = False
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + f"\n[TIMEOUT after {TIMEOUT_SECONDS}s]"
        returncode = -1
        timed_out = True
    duration = time.time() - t0

    score = None
    m = SCORE_RE.search(stdout)
    if m:
        try:
            score = float(m.group(1))
        except ValueError:
            score = None

    return {
        "stdout": truncate_tail(stdout, STDOUT_TRUNC),
        "stderr": truncate_tail(stderr, STDERR_TRUNC),
        "returncode": returncode,
        "score": score,
        "duration_seconds": round(duration, 2),
        "timed_out": timed_out,
    }


# -------------------------- Trace I/O --------------------------

def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_trace(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def write_best(score, iteration):
    BEST_JSON.write_text(
        json.dumps({"score": score, "iteration": iteration}, indent=2),
        encoding="utf-8",
    )


# -------------------------- Prompt builder --------------------------

def recent_iters_summary(trace):
    if not trace:
        return "(no prior iterations)"
    recent = trace[-RECENT_ITERS_IN_PROMPT:][::-1]  # most recent first
    lines = []
    for r in recent:
        plan = (r.get("plan") or "(baseline)").strip().splitlines()[0][:140]
        score = r.get("score")
        score_str = f"{score:.4f}" if isinstance(score, float) else "n/a"
        lines.append(
            f"- iter {r['iteration']} · {r['status']} · score={score_str} · plan: {plan}"
        )
    return "\n".join(lines)


def build_user_prompt(program_md, train_py, best, trace, last_run):
    notes = read_text(NOTES_MD, default="")
    notes_block = truncate_tail(notes, NOTES_TRUNC) if notes.strip() else "empty"

    if best is None:
        best_block = "none yet"
    else:
        best_block = json.dumps(best, indent=2)

    last_stdout = truncate_tail(last_run.get("stdout", ""), STDOUT_TRUNC) if last_run else ""
    last_stderr = truncate_tail(last_run.get("stderr", ""), STDERR_TRUNC) if last_run else ""

    return f"""# Research brief
{program_md}

# Current train.py
```python
{train_py}
```

# Best score so far
{best_block}

# Recent iterations (most recent first)
{recent_iters_summary(trace)}

# Notes file
{notes_block}

# Last run output (if any)
STDOUT (truncated to last {STDOUT_TRUNC} chars):
{last_stdout or '(empty)'}
STDERR (truncated to last {STDERR_TRUNC} chars):
{last_stderr or '(empty)'}

---

Now produce your turn. Use the exact output format from program.md.
"""


# -------------------------- Main loop --------------------------

def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def baseline_iteration(trace_path, run_meta):
    """Run the baseline once, record it as iteration 0."""
    print("→ running baseline...")
    result = execute_train()
    if result["score"] is None:
        raise RuntimeError(
            "Baseline train.py failed. Check workspace/train.py and data/train.csv.\n"
            f"STDERR:\n{result['stderr']}"
        )
    print(f"  baseline score = {result['score']:.4f}")

    record = {
        "iteration": 0,
        "timestamp": now_iso(),
        "duration_seconds": result["duration_seconds"],
        "agent_response_raw": None,
        "reflection": "(baseline — no agent call)",
        "observations": [],
        "hypothesis": None,
        "plan": "baseline logistic regression on raw numeric columns",
        "code": read_text(TRAIN_PY),
        "notes_append": None,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "returncode": result["returncode"],
        "score": result["score"],
        "previous_best": None,
        "kept": True,
        "status": "kept",
        "commit_hash": run_meta["base_commit"],
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    append_jsonl(trace_path, record)
    write_best(result["score"], 0)
    return result["score"], result


def run_loop(max_iterations):
    load_dotenv(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY missing. Put it in .env at the repo root.")

    if not (ROOT / "data" / "train.csv").exists():
        raise RuntimeError(
            "data/train.csv not found. Download from Kaggle and place at data/train.csv."
        )

    if not TRAIN_PY.exists():
        raise RuntimeError(f"{TRAIN_PY} missing. Re-create the baseline first.")

    ensure_clean_repo()
    branch = create_run_branch()
    base_commit = git("rev-parse", "--short", "HEAD").stdout.strip()
    print(f"→ branch: {branch} (base {base_commit})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = TRACES / f"run_{timestamp}.jsonl"
    print(f"→ trace: {trace_path.name}")

    run_meta = {"branch": branch, "base_commit": base_commit, "model": MODEL}

    NOTES_MD.parent.mkdir(parents=True, exist_ok=True)
    if not NOTES_MD.exists():
        NOTES_MD.write_text("", encoding="utf-8")

    best_score, last_run = baseline_iteration(trace_path, run_meta)

    client = anthropic.Anthropic()
    total_cost = 0.0
    iters_without_improvement = 0

    for i in range(1, max_iterations + 1):
        print(f"\n=== iteration {i}/{max_iterations} (best={best_score:.4f}) ===")

        program_md = read_text(PROGRAM_MD)
        train_py = read_text(TRAIN_PY)
        trace = load_trace(trace_path)
        best_obj = json.loads(BEST_JSON.read_text()) if BEST_JSON.exists() else None
        user_prompt = build_user_prompt(program_md, train_py, best_obj, trace, last_run)

        record_base = {
            "iteration": i,
            "timestamp": now_iso(),
            "previous_best": best_score,
        }

        try:
            raw, usage = call_claude(client, user_prompt)
        except Exception as e:
            print(f"  api error: {e}")
            record = {
                **record_base,
                "duration_seconds": 0,
                "agent_response_raw": None,
                "reflection": None, "observations": [], "hypothesis": None,
                "plan": None, "code": None, "notes_append": None,
                "stdout": "", "stderr": f"API error: {e}",
                "returncode": None, "score": None,
                "kept": False, "status": "api_error",
                "commit_hash": None,
                "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            }
            append_jsonl(trace_path, record)
            break

        cost = cost_of(usage) or 0.0
        total_cost += cost
        print(f"  tokens in/out = {usage['input']}/{usage['output']}  cost ≈ ${cost:.4f}  total ≈ ${total_cost:.4f}")

        # Parse
        try:
            parsed = parse_response(raw)
        except ValueError as e:
            print(f"  parse failed: {e}")
            record = {
                **record_base,
                "duration_seconds": 0,
                "agent_response_raw": raw,
                "reflection": None, "observations": [], "hypothesis": None,
                "plan": None, "code": None, "notes_append": None,
                "stdout": "", "stderr": str(e),
                "returncode": None, "score": None,
                "kept": False, "status": "parse_failed",
                "commit_hash": None,
                "input_tokens": usage["input"],
                "output_tokens": usage["output"],
                "cost_usd": cost,
            }
            append_jsonl(trace_path, record)
            iters_without_improvement += 1
            if total_cost >= COST_CAP_USD:
                print(f"→ cost cap hit (${total_cost:.2f}), stopping.")
                break
            continue

        # Write new train.py and append note
        write_text(TRAIN_PY, parsed["code"])
        if parsed["notes_append"]:
            with NOTES_MD.open("a", encoding="utf-8") as f:
                f.write(parsed["notes_append"].strip() + "\n")

        # Commit
        plan_short = parsed["plan"].splitlines()[0][:70]
        commit_sha = git_commit_all(f"iter {i}: {plan_short}")
        if commit_sha is None:
            print("  (no changes to commit — skipping)")

        # Execute
        result = execute_train()
        score = result["score"]
        score_str = f"{score:.4f}" if score is not None else "n/a"
        print(f"  score = {score_str}  ({result['duration_seconds']}s, rc={result['returncode']})")

        # Ratchet
        if score is None:
            status = "crashed"
            kept = False
            if commit_sha:
                git_revert_last()
        elif score < best_score:
            status = "reverted"
            kept = False
            if commit_sha:
                git_revert_last()
        else:
            status = "kept"
            kept = True
            if score > best_score:
                best_score = score
                write_best(score, i)
                iters_without_improvement = 0
            else:
                iters_without_improvement += 1

        if not kept:
            iters_without_improvement += 1

        print(f"  status = {status}")

        record = {
            **record_base,
            "duration_seconds": result["duration_seconds"],
            "agent_response_raw": raw,
            "reflection": parsed["reflection"],
            "observations": parsed["observations_list"],
            "hypothesis": parsed["hypothesis"],
            "plan": parsed["plan"],
            "code": parsed["code"],
            "notes_append": parsed["notes_append"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "returncode": result["returncode"],
            "score": score,
            "kept": kept,
            "status": status,
            "commit_hash": commit_sha if kept else None,
            "input_tokens": usage["input"],
            "output_tokens": usage["output"],
            "cost_usd": cost,
        }
        append_jsonl(trace_path, record)
        last_run = result

        # Stop conditions
        if total_cost >= COST_CAP_USD:
            print(f"→ cost cap hit (${total_cost:.2f}), stopping.")
            break
        if iters_without_improvement >= NO_IMPROVEMENT_LIMIT:
            print(f"→ no improvement for {NO_IMPROVEMENT_LIMIT} iterations, stopping.")
            break

    print(f"\n=== done. best score = {best_score:.4f} · total cost ≈ ${total_cost:.4f} ===")
    print(f"trace: {trace_path}")
    print(f"branch: {branch}")


# -------------------------- CLI --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = ap.parse_args()
    run_loop(max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
