import json
import argparse
import numpy as np
import pandas as pd
import re
import sys
from pathlib import Path
import joblib

from pymorphy3 import MorphAnalyzer
from sentence_transformers import CrossEncoder
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.retrieval.bm25 import load_index, INDEX_PATH
from src.retrieval.retrieve import retrieve_top
from src.retrieval.encoder import rerank_one, MODEL_RERANK, load_chunks_map

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score



def load_jsonl(golden_path):
    rows = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text")
            y = obj.get("target_refuse")
            lang = obj.get("lang", "ru")
            rows.append({
                "id": obj.get("id"),
                "text": f"[{lang}] {text}",
                "y": y,
            })
    return rows


def tune_threshold(y_true, proba):
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.arange(0.10, 0.91, 0.01):
        pred = (proba >= thr).astype(int)
        score = f1_score(y_true, pred, pos_label=1)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return best_thr, best_f1


def main():
    p = argparse.ArgumentParser()    
    p.add_argument("--train", default="/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/sets/refuse_splits/refuse_train.jsonl")
    p.add_argument("--val", default="/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/sets/refuse_splits/refuse_val.jsonl")
    p.add_argument("--test", default="/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/sets/refuse_splits/refuse_test.jsonl")
    p.add_argument("--out", default="/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/crag/action_eval/refuse-logreg.joblib")
    args = p.parse_args()
    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    test_rows = load_jsonl(args.test)
    
    X_train = [r["text"] for r in train_rows]
    y_train = np.array([r["y"] for r in train_rows], dtype=int)
    X_val = [r["text"] for r in val_rows]
    y_val = np.array([r["y"] for r in val_rows], dtype=int)
    X_test = [r["text"] for r in test_rows]
    y_test = np.array([r["y"] for r in test_rows], dtype=int)
    
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    clf.fit(X_train_vec, y_train)

    val_proba = clf.predict_proba(X_val_vec)[:, 1]
    threshold, best_val_f1 = tune_threshold(y_val, val_proba)

    test_proba = clf.predict_proba(X_test_vec)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    print(f"Chosen threshold: {threshold:.2f}")
    print(f"Best val refuse-F1: {best_val_f1:.4f}")
    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, digits=4))
    print("Test confusion matrix:")
    print(confusion_matrix(y_test, test_pred))

    artifact = {
        "vectorizer": vectorizer,
        "model": clf,
        "threshold": threshold,
        "label_positive": "REFUSE",
    }
    joblib.dump(artifact, args.out)
    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()