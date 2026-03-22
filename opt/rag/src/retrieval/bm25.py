import json
import re
import pickle
from razdel import tokenize
from rank_bm25 import BM25Okapi
from pymorphy3 import MorphAnalyzer
from pathlib import Path



N_GRAM_SIZE = 2
CLAUSE_RE = re.compile(r"^\s*(?P<id>\d+(?:\.\d+)*)\s*[\.\)]\s*(?P<body>.+\S)\s*$")
WORD_RE = re.compile(r"^[0-9A-Za-zА-Яа-яЁё]+(?:[.\-/][0-9A-Za-zА-Яа-яЁё]+)*$")


BASE_DIR = Path(__file__).resolve().parents[2]
IN_PATH = BASE_DIR / "data" / "popatkus_all_v5.jsonl"
INDEX_PATH = BASE_DIR / "data" / "bm25index.pkl"

morph = MorphAnalyzer()

def tok(text, lang):
    text = text.lower()
    m = CLAUSE_RE.match(text)
    if m:
        text = m.group("body")
    toks = [t.text for t in tokenize(text)]
    toks = [x for x in toks if WORD_RE.fullmatch(x)]
    if lang == "ru":
        toks = lemma(toks)
    return toks

def lemma(tokens):
    out = []
    for t in tokens:
        if any(ch.isdigit() for ch in t):
            out.append(t)
            continue
        out.append(morph.parse(t)[0].normal_form)
    return out

def add_ngrams(tokens, n=N_GRAM_SIZE):
    out = list(tokens)
    if n >= 2:
        for i in range(len(tokens) - 1):
            out.append(tokens[i] + "_" + tokens[i + 1])
    if n >= 3:
        for i in range(len(tokens) - 2):
            out.append(tokens[i] + "_" + tokens[i + 1] + "_" + tokens[i + 2])
    return out


def load_corpus_by_lang(in_path=IN_PATH, ngram_n=N_GRAM_SIZE):
    ids = {"ru": [], "en": []}
    corpus = {"ru": [], "en": []}
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            lang = obj.get("lang")
            text = obj.get("text")
            cid = obj.get("id")
            terms = add_ngrams(tok(text, lang), n=ngram_n)
            ids[lang].append(cid)
            corpus[lang].append(terms)
    return ids, corpus


def bm25_build(in_path, ngram_n=N_GRAM_SIZE, k1=1.5, b=0.75):
    ids, corpus = load_corpus_by_lang(in_path, ngram_n=ngram_n)
    bm25 = {
        "ru": BM25Okapi(corpus["ru"], k1=k1, b=b) if corpus["ru"] else None,
        "en": BM25Okapi(corpus["en"], k1=k1, b=b) if corpus["en"] else None,
    }
    return bm25, ids


def bm25_search(bm25, ids, query_text, lang="ru", top_k=30, ngram_n=N_GRAM_SIZE):
    model = bm25.get(lang)
    doc_ids = ids.get(lang, [])
    q_terms = add_ngrams(tok(query_text, lang), n=ngram_n)
    scores = model.get_scores(q_terms)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    # return [doc_ids[i] for i in top_idx]
    return [{"id": doc_ids[i], "score": float(scores[i])} for i in top_idx]



def save_index(path, bm25, ids, meta: dict):
    with open(path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "meta": meta}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["bm25"], d["ids"], d.get("meta", {})


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in-path", default=IN_PATH)
    p.add_argument("--index-path", default=INDEX_PATH)
    p.add_argument("--build", action="store_true")
    p.add_argument("--query", default=None)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--ngram-n", type=int, default=2)
    p.add_argument("--lang", choices=["en", "ru"], default="ru")
    p.add_argument("--k1", type=float, default=1.5)
    p.add_argument("--b", type=float, default=0.75)
    args = p.parse_args()

    if args.build:
        bm25, ids = bm25_build(
            args.in_path,
            ngram_n=args.ngram_n,
            k1=args.k1,
            b=args.b
        )
        save_index(
            args.index_path,
            bm25,
            ids,
            {
                "in_path": args.in_path,
                "ngram_n": args.ngram_n,
                "k1": args.k1,
                "b": args.b
            },
        )

    bm25, ids, meta = load_index(args.index_path)
    if args.query:
        ngram_n = meta.get("ngram_n", args.ngram_n)
        res = bm25_search(bm25, ids, args.query, lang=args.lang, top_k=args.top_k, ngram_n=ngram_n)
        for x in res:
            print(x)
