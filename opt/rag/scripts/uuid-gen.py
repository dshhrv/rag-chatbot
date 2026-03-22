import json
import uuid
import re
from pathlib import Path

GOLDEN_SET_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "golden-set")
LANG_RE = re.compile(r"^\s*lang:\s*(?P<body>.+?)\s*$")
Q_RE    = re.compile(r"^\s*q:\s*(?P<body>.+?)\s*$")
REL_RE  = re.compile(r"^\s*rel:\s*(?P<body>.+?)\s*$")
ACTION_RE = re.compile(r"^\s*expected_action:\s*(?P<body>.+?)\s*$")
TARGET_REFUSE_RE = re.compile(r"^\s*target_refuse:\s*(?P<body>.+?)\s*$")

BASE_DIR = Path(__file__).resolve().parents[1]
IN_PATH = BASE_DIR / "data" / "sets" / "evaluation-set.txt"
OUT_PATH = BASE_DIR / "data" / "sets" / "evaluation_golden_set.jsonl"

def dump_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def make_id(*parts, n=24):
    def norm(x):
        if x is None:
            return ""
        if isinstance(x, (list, tuple)):
            return " > ".join(norm(i) for i in x if i not in (None, ""))
        return str(x)
    payload = "|".join(norm(p) for p in parts)
    return str(uuid.uuid5(GOLDEN_SET_NS, payload))


def txt_to_jsonl(in_path=IN_PATH, out_path=OUT_PATH):
    current = {"lang": None, "text": None, "rel": []}
    def flush(fout, cur):
        lang = cur.get("lang")
        text = cur.get("text")
        rel = cur.get("rel")
        expected_action = cur.get("expected_action")
        target_refuse = cur.get("target_refuse")
        obj = {
            "id": make_id(lang, text, rel),
            "lang": lang,
            "text": text,
            "rel": rel,
            "expected_action": expected_action
            # "target_refuse": target_refuse,
        }
        dump_line(fout, obj)
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                flush(fout, current)
                current = {"lang": None, "text": None, "rel": []}
                continue
            m = LANG_RE.match(line)
            if m:
                current["lang"] = m.group("body").strip()
                continue
            m = Q_RE.match(line)
            if m:
                current["text"] = m.group("body").strip()
                continue
            m = REL_RE.match(line)
            if m:
                body = m.group("body")
                rels = [x.strip() for x in body.split(";") if x.strip()]
                current["rel"].extend(rels)
                continue
            m = ACTION_RE.match(line)
            if m:
                current["expected_action"] = m.group("body").strip()
                continue
            m = TARGET_REFUSE_RE.match(line)
            if m:
                current["target_refuse"] = int(m.group("body"))
        flush(fout, current)


if __name__ == "__main__":
    txt_to_jsonl(IN_PATH, OUT_PATH)