from transformers import pipeline
import torch
import time

_model = None
_device = 0 if torch.cuda.is_available() else -1

def get_nli_pipeline():
    global _model
    if _model is None:
        _model = pipeline("text-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=_device, truncation=True, max_length=512)
    return _model

def _safe_extract(output):
    return output if isinstance(output, dict) else output[0]

def verify_answer_simple(premise, hypothesis):
    if not premise.strip() or not hypothesis.strip():
        return {"label": "neutral", "is_hallucination": False, "latency_ms": 0}
    t0 = time.perf_counter()
    pipe = get_nli_pipeline()
    output = pipe({"text": premise, "text_pair": hypothesis})
    result = _safe_extract(output)
    t1 = time.perf_counter()
    label = result["label"]
    is_hallucination = label == "contradiction"
    return {"label": label, "is_hallucination": is_hallucination, "latency_ms": round((t1 - t0) * 1000, 2)}

def verify_answer_with_threshold(premise, hypothesis, entailment_threshold=0.65):
    if not premise.strip() or not hypothesis.strip():
        return {"label": "neutral", "is_hallucination": False, "latency_ms": 0, "score": 0}
    t0 = time.perf_counter()
    pipe = get_nli_pipeline()
    output = pipe({"text": premise, "text_pair": hypothesis})
    result = _safe_extract(output)
    t1 = time.perf_counter()
    label = result["label"]
    score = result["score"]
    is_hallucination = label == "contradiction" or (label == "entailment" and score < entailment_threshold)
    return {"label": label, "score": score, "is_hallucination": is_hallucination, "latency_ms": round((t1 - t0) * 1000, 2)}

def verify_answer_decomposed(premise, hypothesis, entailment_threshold=0.65):
    if not premise.strip() or not hypothesis.strip():
        return {"label": "neutral", "is_hallucination": False, "latency_ms": 0, "per_sentence": []}
    t0 = time.perf_counter()
    pipe = get_nli_pipeline()
    sentences = [s.strip() for s in hypothesis.split(".") if s.strip()]
    per_sentence = []
    for sent in sentences:
        output = pipe({"text": premise, "text_pair": sent})
        r = _safe_extract(output)
        per_sentence.append({"sentence": sent, "label": r["label"], "score": r["score"]})
    is_hallucination = any(r["label"] == "contradiction" for r in per_sentence) or any(r["label"] == "entailment" and r["score"] < entailment_threshold for r in per_sentence)
    t1 = time.perf_counter()
    main_label = "contradiction" if any(r["label"] == "contradiction" for r in per_sentence) else (per_sentence[0]["label"] if per_sentence else "neutral")
    return {"label": main_label, "is_hallucination": is_hallucination, "latency_ms": round((t1 - t0) * 1000, 2), "per_sentence": per_sentence}