from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

from .io.loaders import load_jsonl, read_text_file
from .preprocess.clean import clean_text, high_signal_slice
from .preprocess.chunk import simple_doc_type
from .retrieval.retrieve import build_store
from .llm.client import get_llm
from .llm.prompts import build_prompt
from .schema.validate import enforce_allowed_taxonomy, enforce_schema


def project_root() -> Path:
    # .../repo/src/metaragl/pipeline.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def load_resources():
    root = project_root()
    taxonomy_path = root / "data" / "taxonomy" / "taxonomy.jsonl"
    examples_path = root / "data" / "examples" / "gold_metadata.jsonl"

    taxonomy = load_jsonl(taxonomy_path)
    examples = load_jsonl(examples_path)

    store, embedder = build_store(taxonomy, examples, dim=512)
    allowed_ids = {t["id"] for t in taxonomy}
    return taxonomy, examples, store, embedder, allowed_ids


def safe_json_loads(s: str) -> dict:
    s = (s or "").strip()
    if not s:
        raise ValueError("LLM returned empty output")

    # 1) try direct
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2) strip common code fences
    s2 = re.sub(r"^```(?:json)?\s*|```$", "", s, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(s2)
    except json.JSONDecodeError:
        pass

    # 3) extract first JSON object
    m = re.search(r"\{[\s\S]*\}", s2)
    if m:
        return json.loads(m.group(0))

    raise ValueError(f"Could not find JSON in model output (first 500 chars): {s[:500]}")


def run(doc_path: str) -> Dict[str, Any]:
    taxonomy, examples, store, embedder, allowed_ids = load_resources()

    raw_text = read_text_file(doc_path)
    text = clean_text(raw_text)
    doc_type = simple_doc_type(text)
    excerpt = high_signal_slice(text)

    q_emb = embedder.embed(excerpt)
    tax_hits = [row["meta"] for _, row in store.query(q_emb, rtype="taxonomy", top_k=12)]
    ex_hits = [row["meta"] for _, row in store.query(q_emb, rtype="example", top_k=3)]

    llm = get_llm()
    from .schema.metadata_schema import METADATA_JSON_SCHEMA

    prompt = build_prompt(excerpt, doc_type, tax_hits, ex_hits, METADATA_JSON_SCHEMA)
    raw = llm.generate(prompt)

    obj = safe_json_loads(raw)
    enforce_allowed_taxonomy(obj, allowed_ids)
    record = enforce_schema(obj)
    return record.model_dump()