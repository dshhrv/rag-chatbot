import argparse
import json
from pathlib import Path

import pandas as pd


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_action_accuracy(runs_path):
    runs = load_jsonl(runs_path)
    retrieve_again_count = sum(obj.get("action") == "AGENTIC_ROUTE" for obj in runs)
    print(f"n_retrieve_again: {retrieve_again_count}")
    rows = []
    for obj in runs:
        predicted = obj.get("action")
        expected = obj.get("expected_action")

        if predicted is None or expected is None:
            continue

        rows.append(
            {
                "id": obj.get("id"),
                "lang": obj.get("lang"),
                "query": obj.get("query") or obj.get("text"),
                "expected_action": expected,
                "predicted_action": predicted,
                "correct": int(predicted == expected),
            }
        )

    df = pd.DataFrame(rows)

    total = int(len(df))
    correct = int(df["correct"].sum())
    accuracy = float(correct / total)

    per_label = (df.groupby("expected_action", as_index=False).agg(support=("id", "count"), correct=("correct", "sum")).sort_values("expected_action").reset_index(drop=True))
    per_label["accuracy"] = per_label["correct"] / per_label["support"]
    per_label["accuracy_percent"] = (per_label["accuracy"] * 100).round(2)
    wrong_df = df[df["correct"] == 0].copy()
    return df, per_label, wrong_df, total, correct, accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/crag/crag_out.jsonl")
    p.add_argument("--out-dir", default="/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/crag/action_eval")
    args = p.parse_args()
    df, per_label, wrong_df, total, correct, accuracy = evaluate_action_accuracy(args.runs)

    print(f"n_examples: {total}")
    print(f"n_correct: {correct}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"accuracy_percent: {accuracy * 100.0:.2f}%")
    print()
    print("per_label:")
    print(per_label.to_string(index=False))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(out_dir / "action_predictions.csv", index=False)
        per_label.to_csv(out_dir / "action_accuracy_by_label.csv", index=False)
        wrong_df.to_csv(out_dir / "action_errors.csv", index=False)


if __name__ == "__main__":
    main()