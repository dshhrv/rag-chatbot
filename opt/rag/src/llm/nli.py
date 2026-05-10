import re
import time

import torch
from transformers import pipeline


MODEL_NAME = "MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli"

_model = None
_device = 0 if torch.cuda.is_available() else -1


LABEL_ALIASES = {
    "entailment": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
    "entails": "entailment",
    "not_entailment": "neutral",
    "not entailment": "neutral",
}


def get_nli_pipeline():
    global _model
    if _model is None:
        _model = pipeline(
            "text-classification",
            model=MODEL_NAME,
            device=_device,
            truncation=True,
            max_length=512,
        )
    return _model


def _normalize_label(label):
    label = str(label).strip().lower()
    label = label.replace("label_", "label_")
    return LABEL_ALIASES.get(label, label)


def _flatten_pipeline_output(output):
    if isinstance(output, dict):
        return [output]

    if isinstance(output, list):
        if not output:
            return []
        if isinstance(output[0], dict):
            return output
        if isinstance(output[0], list):
            return output[0]

    return []


def _run_nli(premise, hypothesis):
    premise = (premise or "").strip()
    hypothesis = (hypothesis or "").strip()

    if not premise or not hypothesis:
        return {
            "label": "neutral",
            "score": 0.0,
            "scores": {
                "entailment": 0.0,
                "neutral": 1.0,
                "contradiction": 0.0,
            },
            "latency_ms": 0.0,
        }

    pipe = get_nli_pipeline()

    t0 = time.perf_counter()
    try:
        output = pipe(
            {
                "text": premise,
                "text_pair": hypothesis,
            },
            top_k=None,
        )
    except TypeError:
        output = pipe(
            {
                "text": premise,
                "text_pair": hypothesis,
            },
            return_all_scores=True,
        )
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    rows = _flatten_pipeline_output(output)
    scores = {}

    for row in rows:
        label = _normalize_label(row.get("label", "neutral"))
        try:
            score = float(row.get("score", 0.0))
        except Exception:
            score = 0.0
        scores[label] = score

    for label in ["entailment", "neutral", "contradiction"]:
        scores.setdefault(label, 0.0)

    label = max(scores, key=scores.get)
    score = scores[label]

    return {
        "label": label,
        "score": round(score, 6),
        "scores": {
            k: round(v, 6)
            for k, v in scores.items()
        },
        "latency_ms": latency_ms,
    }


def split_sentences(text):
    text = (text or "").strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    sentences = [
        p.strip()
        for p in parts
        if p and p.strip()
    ]

    if len(sentences) <= 1:
        sentences = [
            p.strip()
            for p in re.split(r"[.!?]+", text)
            if p.strip()
        ]

    return sentences or [text]


def verify_answer_simple(premise, hypothesis):
    result = _run_nli(premise, hypothesis)
    label = result["label"]
    result["is_hallucination"] = label == "contradiction"
    result["method"] = "simple"
    return result


def verify_answer_with_threshold(
    premise,
    hypothesis,
    entailment_threshold=0.65,
    contradiction_threshold=0.50,
):
    result = _run_nli(premise, hypothesis)
    scores = result.get("scores", {})
    entailment_score = float(scores.get("entailment", 0.0))
    contradiction_score = float(scores.get("contradiction", 0.0))

    result["is_hallucination"] = (
        contradiction_score >= contradiction_threshold
        or entailment_score < entailment_threshold
    )
    result["entailment_threshold"] = entailment_threshold
    result["contradiction_threshold"] = contradiction_threshold
    result["method"] = "threshold"
    return result


def verify_answer_decomposed(
    premise,
    hypothesis,
    entailment_threshold=0.65,
    contradiction_threshold=0.50,
):
    premise = (premise or "").strip()
    hypothesis = (hypothesis or "").strip()

    if not premise or not hypothesis:
        return {
            "label": "neutral",
            "score": 0.0,
            "scores": {
                "entailment": 0.0,
                "neutral": 1.0,
                "contradiction": 0.0,
            },
            "is_hallucination": False,
            "latency_ms": 0.0,
            "method": "decomposed",
            "per_sentence": [],
            "n_sentences": 0,
        }

    sentences = split_sentences(hypothesis)
    per_sentence = []

    t0 = time.perf_counter()
    for idx, sent in enumerate(sentences, 1):
        r = verify_answer_with_threshold(
            premise,
            sent,
            entailment_threshold=entailment_threshold,
            contradiction_threshold=contradiction_threshold,
        )
        per_sentence.append(
            {
                "sentence_id": idx,
                "sentence": sent,
                "label": r["label"],
                "score": r.get("score", 0.0),
                "entailment_score": r.get("scores", {}).get("entailment", 0.0),
                "neutral_score": r.get("scores", {}).get("neutral", 0.0),
                "contradiction_score": r.get("scores", {}).get("contradiction", 0.0),
                "is_hallucination": r.get("is_hallucination", False),
                "latency_ms": r.get("latency_ms", 0.0),
            }
        )
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    contradiction_count = sum(
        1
        for r in per_sentence
        if r["label"] == "contradiction"
    )
    neutral_count = sum(
        1
        for r in per_sentence
        if r["label"] == "neutral"
    )
    hallucination_count = sum(
        1
        for r in per_sentence
        if r["is_hallucination"]
    )

    if contradiction_count > 0:
        label = "contradiction"
    elif neutral_count > 0:
        label = "neutral"
    else:
        label = "entailment"

    mean_scores = {
        "entailment": sum(r["entailment_score"] for r in per_sentence) / len(per_sentence),
        "neutral": sum(r["neutral_score"] for r in per_sentence) / len(per_sentence),
        "contradiction": sum(r["contradiction_score"] for r in per_sentence) / len(per_sentence),
    }

    return {
        "label": label,
        "score": round(max(mean_scores.values()), 6),
        "scores": {
            k: round(v, 6)
            for k, v in mean_scores.items()
        },
        "is_hallucination": hallucination_count > 0,
        "latency_ms": latency_ms,
        "method": "decomposed",
        "per_sentence": per_sentence,
        "n_sentences": len(per_sentence),
        "contradiction_sentence_count": contradiction_count,
        "neutral_sentence_count": neutral_count,
        "hallucination_sentence_count": hallucination_count,
    }
