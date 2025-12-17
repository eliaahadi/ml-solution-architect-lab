# rag/retriever.py
from __future__ import annotations
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from rag.ingest import Chunk

INDEX_DIR = Path("rag_index")

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = faiss.read_index(str(INDEX_DIR / "docs.faiss"))
        with open(INDEX_DIR / "chunks.pkl", "rb") as f:
            self.chunks: List[Chunk] = pickle.load(f)

    def search(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        q = self.model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        scores, idxs = self.index.search(q, top_k)
        results: List[Tuple[Chunk, float]] = []
        for i, score in zip(idxs[0].tolist(), scores[0].tolist()):
            if i < 0:
                continue
            results.append((self.chunks[i], float(score)))
        return results