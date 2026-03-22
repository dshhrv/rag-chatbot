import json
import docx
import re
from docx.table import Table, _Cell
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from hashlib import sha1
from docx.text.paragraph import Paragraph
from razdel import sentenize
from zipfile import ZipFile
from lxml import etree
import uuid


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}

DOCX_PATH_RU = "/opt/rag/data/popatkus_ru_ready.docx"
OUT_PATH_RU = "/opt/rag/data/popatkus_ru_v0.jsonl"
DOCX_PATH_EN = "/opt/rag/data/popatkus_en_ready.docx"
OUT_PATH_EN = "/opt/rag/data/popatkus_en_v0.jsonl"
DOC_ID_RU = "popatkus_ru"
DOC_ID_EN = "popatkus_en"
LANG_RU = "ru"
LANG_EN = "en"
OUT_PATH_ALL = "popatkus_all_v5.jsonl"
GLOSS_RE = re.compile(r"^\s*(?P<term>.{3,80}?)\s+(?:[—–-]|refer\s+to|refers\s+to)\s+(?P<def>.+\S)\s*$")
TERM_RE = re.compile(r"([A-ZА-ЯЁ])+.+")
CLAUSE_RE = re.compile(r"^\s*(?P<id>\d+(?:\.\d+)*)\s*[\.\)]\s*(?P<body>.+\S)\s*$")
SECTION_RE = re.compile(r"^\s*(?P<num>\d{1,2})\.\s+(?P<title>.+\S)\s*$")
QDRANT_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "popatkus")
MAX_CHARS = 1000
OVERLAP = 120
path = OUT_PATH_RU



def load_footnotes(doc_path):
    with ZipFile(doc_path) as zip:
        root = etree.fromstring(zip.read("word/footnotes.xml"))
    m = {}
    for fn in root.xpath(".//w:footnote", namespaces=NS):
        fid = int(fn.get(f"{{{W}}}id"))
        if fid <= 0:
            continue
        m[fid] = "".join(fn.xpath(".//w:t/text()", namespaces=NS)).strip()
    return m


def footnote_ids_in_paragraph(p):
    return [int(x) for x in p._p.xpath(".//w:footnoteReference/@w:id")]


def clean(s):
    return (s or "").replace("\u00A0", " ").strip()


def iter_children(parent):
    if hasattr(parent, 'element') and hasattr(parent.element, 'body'):
        parent_elm = parent.element.body
        parent_obj = parent
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
        parent_obj = parent
    elif hasattr(parent, "_element"):
        parent_elm = parent._element
        parent_obj = parent
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent_obj)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent_obj)
        else:
            continue


def table_to_text(tbl):
    lines = []
    for row in tbl.rows:
        cells = []
        for cell in row.cells:
            cell_txt = " ".join(clean(p.text) for p in cell.paragraphs if clean(p.text))
            cells.append(clean(cell_txt))
        line = clean(" | ".join([c for c in cells if c]))
        if line:
            lines.append(line)
    return "\n".join(lines)


def make_id(*parts, n=24):
    def norm(x):
        if x is None:
            return ""
        if isinstance(x, (list, tuple)):
            return " > ".join(norm(i) for i in x if i not in (None, ""))
        return str(x)

    payload = "|".join(norm(p) for p in parts)
    return str(uuid.uuid5(QDRANT_NS, payload))


def split_sentences(s):
    s = s.replace("\n", " ").strip()
    return [x.text.strip() for x in sentenize(s) if x.text.strip()]


def bold_text(s: Paragraph):
    runs = [runs for runs in s.runs if clean(runs.text)]
    if not runs:
        return False
    return all(run.bold is True for run in runs)


def is_section_heading(p):
    txt = clean(p.text)
    s = SECTION_RE.match(txt)
    if not s:
        return False
    n = int(s.group("num"))
    if not (1 <= n <= 20):
        return False
    return bold_text(p)


def nearest_container_prefix(clause_id, prefix_by_id):
    while "." in clause_id:
        clause_id = clause_id.rsplit(".", 1)[0]
        if clause_id in prefix_by_id:
            return prefix_by_id[clause_id]
    return None


def dump_line(f, obj: dict):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def export_jsonl(docx_path, out_path, doc_id, lang, type="w"):
    footnote_dict = load_footnotes(docx_path)
    with open(out_path, type, encoding="utf-8") as f:
        prefix_by_id = {}
        doc = docx.Document(docx_path)
        heading_stack = []
        current = None

        def start_clause(clause_id, cleaned_text, definition, term=None, heading_path=None, type=None, footnote=None):
            nonlocal current
            hp = list(heading_path) if heading_path else []
            current = {
                "id": make_id(doc_id, lang, type, hp, clause_id, term, cleaned_text),
                "clause_id": clause_id,
                "lang": lang,
                "type": type,
                "term": term,
                "definition": [] if definition is None else [clean(definition)],
                "text_parts": [cleaned_text],
                "heading_path": hp,
                "footnotes": [] if not footnote else [clean(x) for x in footnote]
            }
            return current

        def split_long_piece(s, max_chars):
            s = clean(s).replace("\n", " ").strip()
            if not s:
                return []
            parts = [s]
            for pat in (r"(?<=[;:])\s+", r"(?<=,)\s+"):
                new_parts = []
                for p in parts:
                    if len(p) <= max_chars:
                        new_parts.append(p)
                    else:
                        new_parts.extend([x.strip() for x in re.split(pat, p) if x.strip()])
                parts = new_parts
            packed = []
            buf = ""
            for p in parts:
                if not buf:
                    buf = p
                elif len(buf) + 1 + len(p) <= max_chars:
                    buf = buf + " " + p
                else:
                    packed.append(buf.strip())
                    buf = p
            if buf:
                packed.append(buf.strip())
            final = []
            for p in packed:
                if len(p) <= max_chars:
                    final.append(p)
                    continue
                start = 0
                while start < len(p):
                    end = min(start + max_chars, len(p))
                    if end < len(p):
                        ws = p.rfind(" ", start, end)
                        if ws > start + int(max_chars * 0.6):
                            end = ws
                    final.append(p[start:end].strip())
                    start = end
            return [x for x in final if x]

        def flush():
            nonlocal current
            if current is None:
                return
            hp = current["heading_path"]
            text = "\n".join(current["text_parts"])
            definition = "\n".join(current["definition"])
            footnotes = current["footnotes"]

            def dump_rule_chunk(chunk_text, chunk_index):
                obj = {
                    "id": make_id(doc_id, lang, current["type"], hp, current["clause_id"], current["term"], chunk_text,
                                  chunk_index),
                    "chunk_index": chunk_index,
                    "doc_id": doc_id,
                    "clause_id": current["clause_id"],
                    "lang": lang,
                    "type": current["type"],
                    "text": chunk_text,
                    "heading_path": hp,
                    "meta": {
                        "source_file": docx_path,
                        "version": out_path,
                        "footnotes": footnotes,
                    }
                }
                dump_line(f, obj)

            if current["type"] == "glossary":
                obj = {
                    "id": current["id"],
                    "doc_id": doc_id,
                    "lang": lang,
                    "type": current["type"],
                    "heading_path": hp,
                    "text": text,
                    "meta": {
                        "source_file": docx_path,
                        "term": current["term"],
                        "definition": definition,
                        "version": out_path,
                        "footnotes": footnotes,
                    }
                }
                dump_line(f, obj)
                current = None
                return
            if len(text) <= MAX_CHARS:
                obj = {
                    "id": current["id"],
                    "doc_id": doc_id,
                    "clause_id": current["clause_id"],
                    "lang": lang,
                    "type": current["type"],
                    "text": text,
                    "heading_path": hp,
                    "meta": {
                        "source_file": docx_path,
                        "version": out_path,
                        "footnotes": footnotes,
                    }
                }
                dump_line(f, obj)
                current = None
                return
            parts = []
            for s in split_sentences(text):
                if len(s) <= MAX_CHARS:
                    parts.append(s)
                else:
                    parts.extend(split_long_piece(s, MAX_CHARS))
            chunk_index = 1
            chunk_text = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if not chunk_text:
                    chunk_text = part
                    continue
                candidate = chunk_text + " " + part
                if len(candidate) <= MAX_CHARS:
                    chunk_text = candidate
                    continue
                dump_rule_chunk(chunk_text.strip(), chunk_index)
                chunk_index += 1
                carry = chunk_text[-OVERLAP:].lstrip() if OVERLAP > 0 else ""
                if carry and len(carry) + 1 + len(part) > MAX_CHARS:
                    carry = ""
                chunk_text = (carry + " " + part).strip() if carry else part
            if chunk_text.strip():
                dump_rule_chunk(chunk_text.strip(), chunk_index)
            current = None

        for child in iter_children(doc):
            if isinstance(child, Paragraph):
                if child.style.name == "Heading 1":
                    flush()
                    heading_stack = [clean(child.text)]
                    continue
                elif child.style.name == "Normal":
                    ids = footnote_ids_in_paragraph(child)
                    if is_section_heading(child):
                        flush()
                        heading_stack = heading_stack[:1] + [child.text]
                        continue
                    m = GLOSS_RE.match(child.text)
                    if m and TERM_RE.match(m.group("term")):
                        d = clean(m.group("term"))
                        heading_stack = ["Glossary", f"{d}"]
                        flush()
                        start_clause(None,
                                     cleaned_text=clean(child.text),
                                     term=clean(m.group("term")),
                                     definition=clean(m.group("def")),
                                     type="glossary",
                                     heading_path=heading_stack.copy(),
                                     footnote=[footnote_dict[i] for i in ids if i in footnote_dict]
                                     )
                        continue
                    t = CLAUSE_RE.match(child.text)
                    if t:
                        cid = t.group("id")
                        body = clean(t.group("body"))
                        if body.endswith(":"):
                            flush()
                            prefix_by_id[cid] = f"{cid}. {body}"
                            continue
                        prefix = nearest_container_prefix(cid, prefix_by_id)
                        full_text = f"{cid}. {body}" if not prefix else f"{prefix}\n{cid}. {body}"
                        flush()
                        start_clause(cid,
                                     full_text,
                                     term=None,
                                     definition=None,
                                     type="rules",
                                     heading_path=heading_stack.copy(),
                                     footnote=[footnote_dict[i] for i in ids if i in footnote_dict]
                                     )
                        continue
                    if current is not None:
                        current["footnotes"].extend(footnote_dict[i] for i in ids if i in footnote_dict)
                        current["text_parts"].append(clean(child.text))
                        if current["type"] == "glossary":
                            current["definition"].append(clean(child.text))
                    continue
            if isinstance(child, Table):
                tbl = table_to_text(child)
                if current is not None:
                    current["text_parts"].append(tbl)
        flush()


if __name__ == "__main__":
    export_jsonl(
        docx_path=DOCX_PATH_RU,
        out_path=path,
        doc_id=DOC_ID_RU,
        lang=LANG_RU,
        type="w"
    )
    # export_jsonl(
    #     docx_path=DOCX_PATH_EN,
    #     out_path=path,
    #     doc_id=DOC_ID_EN,
    #     lang=LANG_EN,
    #     type="a"
    # )
