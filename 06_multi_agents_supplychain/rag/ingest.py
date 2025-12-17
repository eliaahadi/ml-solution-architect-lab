# rag/ingest.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

@dataclass
class Chunk:
    doc_id: str
    source: str
    chunk_id: int
    text: str

def _chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    # paragraph-aware chunking, then window by chars
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    joined = "\n\n".join(paras)

    chunks: List[str] = []
    start = 0
    n = len(joined)
    while start < n:
        end = min(n, start + max_chars)
        chunk = joined[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, end - overlap)
        if end == n:
            break
    return chunks

def ingest_docs(docs_dir: str = "docs") -> List[Chunk]:
    base = Path(docs_dir)
    files = sorted([p for p in base.glob("**/*") if p.is_file() and p.suffix.lower() in {".md", ".txt"}])

    all_chunks: List[Chunk] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        doc_id = f.stem
        pieces = _chunk_text(text)
        for i, piece in enumerate(pieces):
            all_chunks.append(Chunk(
                doc_id=doc_id,
                source=str(f),
                chunk_id=i,
                text=piece,
            ))
    return all_chunks