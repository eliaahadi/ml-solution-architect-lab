from typing import Dict, List, Any, Tuple
import math

def cosine(a: List[float], b: List[float]) -> float:
    return sum(x*y for x, y in zip(a, b))

class InMemoryVectorStore:
    """
    Small-scale store for local testing. For production you'd swap in FAISS/pgvector/etc.
    """
    def __init__(self):
        self._rows: List[Dict[str, Any]] = []  # {id, type, text, meta, emb}

    def add(self, rid: str, rtype: str, text: str, meta: Dict[str, Any], emb: List[float]):
        self._rows.append({"id": rid, "type": rtype, "text": text, "meta": meta, "emb": emb})

    def query(self, q_emb: List[float], rtype: str, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
        scored = []
        for row in self._rows:
            if row["type"] != rtype:
                continue
            s = cosine(q_emb, row["emb"])
            scored.append((s, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]