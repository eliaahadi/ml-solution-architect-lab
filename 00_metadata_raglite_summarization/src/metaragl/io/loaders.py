from pathlib import Path
import json

def load_jsonl(path: str):
    items = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing: {path}")
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items

def read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")