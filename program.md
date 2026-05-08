# program.md

## Problem

You are an ML research agent working on the Spaceship Titanic Kaggle competition. Binary classification: predict whether each passenger was transported to an alternate dimension (column: `Transported`).

Training data: `./data/train.csv` (~8,700 rows).
The target column is `Transported` (boolean).
A held-out validation split is handled inside `prepare.py`. You must not modify that file.

## Your job

Modify `train.py` to improve validation accuracy. Each turn, propose one concrete change, implement it, and run the script. Read the result, decide what to do next, repeat.

## Baseline to beat

A logistic regression on the raw numeric columns (with simple imputation) achieves around **0.78 validation accuracy**. That is the number to beat. The current best score is tracked in `best.json` — read it before proposing changes.

## Hard rules

- The script `train.py` must call `prepare.evaluate(predict_fn)` exactly once and print the returned accuracy as the last line of stdout in the format `VAL_ACCURACY: 0.XXXX`. Anything else and your run will be discarded.
- The script must complete in under 60 seconds. If it doesn't, the run is killed and reverted.
- You may use only these libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`. No internet. No installing new packages.
- You may not modify `prepare.py`. The evaluator is fixed.
- You may not look at the test set. There is no test set in your environment.

## Soft directives

**Simpler is better.** All else being equal, a small improvement that adds ugly complexity is not worth it. A 30-line clean solution is better than a 200-line ensemble even if the ensemble scores marginally higher.

**Look at the data before modelling it.** The first useful thing you can do is understand what's in front of you. Print summaries, check distributions, look for missing values, find correlations. The features in this dataset have structure that rewards inspection.

**If you find yourself trying minor variants of the same idea, stop.** Three iterations of XGBoost with slightly different hyperparameters is a sign you're stuck. Try something structurally different — different features, different model class, different way of handling missingness.

**Keep notes.** Maintain a file called `notes.md` in the working directory. After each iteration, append one line: what you tried, what you learned. Read it back at the start of each turn so you don't repeat yourself.

**Refactor when the file grows.** Each turn you must rewrite the full contents of `train.py`, so the file's size directly costs output tokens. The harness shows you the current size and budget at the top of each prompt. If you are approaching the budget — or the file is becoming hard to reason about — make this turn a behavior-preserving refactor: extract helpers, remove dead code, consolidate duplicated transforms. A refactor that ties the current best score is fine; the ratchet keeps it. Only refactor when needed, not as filler.

**Track your best.** Maintain `best.json` with the highest score you've achieved and the iteration that produced it. Update it only when you beat the current best.

## Hints (without solutions)

These are areas where the data has structure worth investigating. They are not instructions for what to try — they are directions to point your attention.

- Some columns encode multiple pieces of information in one string. They may be more useful split apart than left whole.
- Some columns are correlated with each other in ways that aren't immediately obvious from the column descriptions. Look at conditional distributions.
- Identifier columns can encode group structure. Don't dismiss them as "just IDs" without checking.
- Missing values may not be missing at random. The fact that a value is missing can itself be predictive.
- The dataset is small enough that you can afford to look at it manually. Use `df.head()`, `df.describe()`, `df.groupby().agg()` liberally.

## How to handle failures

- If your code crashes, read the error message carefully. Don't blindly retry — fix the cause.
- If a typo or import error caused the failure, fix it and re-run.
- If the idea is broken at the root (e.g., a feature you assumed exists doesn't), abandon the idea and try something else. Don't sink three iterations into rescuing a doomed hypothesis.
- If your score regresses badly, the change will be reverted automatically by the harness. Read the revert as feedback: that direction didn't work, try something else.

## Loop discipline

**NEVER STOP.** Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask for permission, clarification, or approval. The human has set the direction in this file; your job is to execute. If you are uncertain, make your best guess and run the experiment — the result will tell you whether you were right.

If you genuinely believe the problem is solved (i.e., you are confident no further change you can think of would help), state that explicitly in your response and the harness will end the run. But this should be rare; treat the budget as something to use, not preserve.

## Output format per turn

Each of your responses must use the following XML-tag structure. The harness parses these tags exactly — missing or malformed tags cause the iteration to be discarded.

```
<reflection>
2–3 sentences: what did the last result tell you? What's your current best
understanding of the problem?
</reflection>

<observations>
3–5 bullet points of specific, concrete observations about the data, the model
behaviour, or the trace so far. Not generic ML platitudes — grounded notes
from staring at this dataset. Each bullet is one short paragraph (1–3 sentences).
</observations>

<hypothesis>
1–2 sentences: a falsifiable claim the next experiment will test.
e.g. "Splitting Cabin into deck/num/side will improve accuracy because deck
position correlates with the target."
</hypothesis>

<plan>
1–2 sentences: the concrete change you will make this turn, and why.
</plan>

<code>
```python
# full new contents of train.py
```
</code>

<notes_append>
One line to append to notes.md.
</notes_append>
```

Do not include anything outside these six tags. No preamble, no closing pleasantries, no "let me know if you want me to continue." Just the six tagged sections.
