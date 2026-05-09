"""
submit.py — Generate a Kaggle submission from the agent's best train.py.

One-off script. NOT part of the autoresearch loop and lives outside the
agent's contract.

Pipeline:
  1. Import workspace/train.py and reproduce the agent's 80/20 held-out
     validation score as a sanity check.
  2. Retrain the same model on the FULL labelled set (all rows of
     data/train.csv), so the submitted model has seen every available
     example.
  3. Predict on data/test.csv, validate the format against
     data/sample_submission.csv, and write submission.csv.

Run after placing test.csv and sample_submission.csv at data/:
    python submit.py
Then upload submission.csv at
https://www.kaggle.com/competitions/spaceship-titanic/submit
"""
import sys
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"
SUBMISSION = ROOT / "submission.csv"

# Make workspace/train.py importable. (workspace/train.py itself adds the
# repo root to sys.path on import, so `import prepare` will resolve too.)
sys.path.insert(0, str(ROOT / "workspace"))


def main():
    missing = [p.name for p in (TEST_CSV, SAMPLE_SUB) if not p.exists()]
    if missing:
        sys.exit(
            f"Place test.csv and sample_submission.csv in {DATA_DIR}/ — see README. "
            f"Missing: {', '.join(missing)}"
        )
    if not TRAIN_CSV.exists():
        sys.exit(f"Missing {TRAIN_CSV}. Place the Kaggle train.csv there.")

    import prepare         # noqa: E402 — sys.path modified above
    import train as agent  # workspace/train.py

    # Step 1: sanity — reproduce the agent's held-out validation score
    # using the unmodified 80/20 split. Confirms our import is wired up
    # to exactly the model the agent reported.
    print("[1/3] Training on 80% fold (sanity check)...")
    predict_fn_split = agent.build_predict_fn()
    val_score = prepare.evaluate(predict_fn_split)
    print(f"      held-out validation score: {val_score:.4f}")

    # Step 2: retrain on the full labelled set. We monkey-patch
    # prepare.load_split so build_predict_fn fits on every row. The
    # second tuple element is an empty frame because build_predict_fn
    # only reads train_df, not val_df. Restored immediately after.
    full_df = pd.read_csv(prepare.DATA_PATH)
    original_load_split = prepare.load_split
    prepare.load_split = lambda: (
        full_df.reset_index(drop=True),
        full_df.iloc[:0].reset_index(drop=True),
    )
    try:
        print(f"\n[2/3] Retraining on full set ({len(full_df)} rows)...")
        predict_fn_full = agent.build_predict_fn()
    finally:
        prepare.load_split = original_load_split

    # Step 3: predict, validate, write.
    test_df = pd.read_csv(TEST_CSV)
    sample_df = pd.read_csv(SAMPLE_SUB)
    print(f"\n[3/3] Predicting on {len(test_df)} test rows...")
    preds = predict_fn_full(test_df)

    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"].values,
        "Transported": pd.Series(preds).astype(bool).values,
    })
    # Match sample_submission's row ordering by PassengerId so the file is
    # byte-comparable to the sample apart from the Transported column.
    submission = (
        submission.set_index("PassengerId")
                  .reindex(sample_df["PassengerId"])
                  .reset_index()
    )

    if len(submission) != len(sample_df):
        sys.exit(
            f"row count mismatch: submission has {len(submission)}, "
            f"sample has {len(sample_df)}"
        )
    if set(submission["PassengerId"]) != set(sample_df["PassengerId"]):
        sys.exit("PassengerId set mismatch between submission and sample_submission.csv")
    if submission["Transported"].isna().any():
        sys.exit("Submission contains NaN — a test PassengerId was not predicted")

    submission.to_csv(SUBMISSION, index=False)

    n_true = int(submission["Transported"].sum())
    n_false = len(submission) - n_true
    print(f"\n=== submission.csv written ({SUBMISSION.name}) ===")
    print(f"  rows:                       {len(submission)}")
    print(f"  Transported = True:         {n_true} ({n_true / len(submission) * 100:.1f}%)")
    print(f"  Transported = False:        {n_false} ({n_false / len(submission) * 100:.1f}%)")
    print(f"  held-out validation score:  {val_score:.4f}")


if __name__ == "__main__":
    main()
