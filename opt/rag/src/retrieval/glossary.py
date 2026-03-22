import json
import re


IN_PATH = "/opt/rag/data/popatkus_all_v5.jsonl"
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", re.UNICODE)

terms_ru = {}
terms_en = {}

def make_dict(in_path=IN_PATH):
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj["type"] == "glossary":
                m = obj.get("meta", {})
                term = m.get("term", "")
                definition = m.get("definition", "")
                lang = obj.get("lang")
                if lang == "ru":
                    terms_ru[term.lower().strip()] = definition.strip()
                elif lang == "en":
                    terms_en[term.lower().strip()] = definition.strip()
                
            

def detect_terms(text, lang):
    d = terms_ru if lang == "ru" else terms_en
    if text:
        text_norm = text.lower().strip()
    text_norm = text
    found = set()
    for t in d.keys():
        if " " in t:
            if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text_norm):
                found.add(t)
    for w in WORD_RE.findall(text_norm):
        if w in d:
            found.add(w)
    return list(found)


def get_definitions(term, lang):
    d = terms_ru if lang == "ru" else terms_en
    return d.get(term.lower().strip())


def format_definitions(terms_list, lang, max_n=5):
    out = []
    for t in terms_list[:max_n]:
        definition = get_definitions(t, lang)
        if lang == "ru":
            out.append(f"{t} — {definition}")
        else:
            out.append(f"{t} refers to {definition}")
    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-path", default=IN_PATH)
    parser.add_argument("--lang", choices=["ru", "en"], default="ru")
    parser.add_argument("--query", default=None)
    parser.add_argument("--max-n", type=int, default=5)
    args = parser.parse_args()
    
    
    make_dict(args.in_path)
    if args.query:
        hits = detect_terms(args.query, args.lang)
        defs = format_definitions(hits, args.lang, max_n=args.max_n)
        for x in defs:
            print(x)
    else:
        d = terms_ru if args.lang == "ru" else terms_en
        print(len(d))

