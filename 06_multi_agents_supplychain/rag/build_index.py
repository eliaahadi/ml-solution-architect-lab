# rag/build_index.py
from __future__ import annotations

import pickle
import sys
from pathlib import Path

# Allow running as a script from repo root: `python rag/build_index.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from rag.ingest import ingest_docs

INDEX_DIR = Path("rag_index")
INDEX_DIR.mkdir(exist_ok=True)

def main():
    chunks = ingest_docs("docs")
    if not chunks:
        raise SystemExit("No docs found in ./docs (add .md or .txt files).")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [c.text for c in chunks]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine similarity if embeddings normalized
    index.add(emb)

    faiss.write_index(index, str(INDEX_DIR / "docs.faiss"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Built index with {len(chunks)} chunks into {INDEX_DIR}/")

if __name__ == "__main__":
    main()