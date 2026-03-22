import json
from pathlib import Path
from sklearn.model_selection import train_test_split

IN_PATH = Path("/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/sets/refuse_set.jsonl")
OUT_DIR = Path("/home/dshhrv666/server1/popatkus-rag-bot/opt/rag/data/sets/refuse_splits/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def dump_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

rows = load_jsonl(IN_PATH)

rows = [r for r in rows if "target_refuse" in r]

y = [int(r["target_refuse"]) for r in rows]

train_rows, temp_rows = train_test_split(
    rows,
    test_size=0.30,
    random_state=42,
    shuffle=True,
    stratify=y,
)

y_temp = [int(r["target_refuse"]) for r in temp_rows]

val_rows, test_rows = train_test_split(
    temp_rows,
    test_size=0.50,
    random_state=42,
    shuffle=True,
    stratify=y_temp,
)

dump_jsonl(OUT_DIR / "refuse_train.jsonl", train_rows)
dump_jsonl(OUT_DIR / "refuse_val.jsonl", val_rows)
dump_jsonl(OUT_DIR / "refuse_test.jsonl", test_rows)

print(f"total={len(rows)}")
print(f"train={len(train_rows)}")
print(f"val={len(val_rows)}")
print(f"test={len(test_rows)}")

for name, part in [
    ("train", train_rows),
    ("val", val_rows),
    ("test", test_rows),
]:
    pos = sum(int(r["target_refuse"]) for r in part)
    neg = len(part) - pos
    print(f"{name}: refuse={pos}, not_refuse={neg}")