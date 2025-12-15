from typing import List, Dict, Any
from .embedder import HashEmbedder
from .vector_store import InMemoryVectorStore

def record_to_text(rec: Dict[str, Any]) -> str:
    # index-able representation
    parts = []
    if "label" in rec: parts.append(rec["label"])
    if "description" in rec: parts.append(rec["description"])
    if "synonyms" in rec: parts.append(" ".join(rec["synonyms"]))
    if "title" in rec: parts.append(rec["title"])
    if "abstract" in rec: parts.append(rec["abstract"])
    if "keywords" in rec: parts.append(" ".join(rec["keywords"]))
    return " | ".join([p for p in parts if p])

def build_store(taxonomy: List[Dict[str, Any]], examples: List[Dict[str, Any]], dim: int = 512):
    emb = HashEmbedder(dim=dim)
    store = InMemoryVectorStore()

    for t in taxonomy:
        store.add(
            rid=t["id"],
            rtype="taxonomy",
            text=record_to_text(t),
            meta=t,
            emb=emb.embed(record_to_text(t))
        )

    for ex in examples:
        store.add(
            rid=ex["id"],
            rtype="example",
            text=record_to_text(ex),
            meta=ex,
            emb=emb.embed(record_to_text(ex))
        )

    return store, emb