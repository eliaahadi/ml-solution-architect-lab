import json
from typing import List, Dict, Any

def build_prompt(doc_text: str,
                 doc_type: str,
                 taxonomy_hits: List[Dict[str, Any]],
                 example_hits: List[Dict[str, Any]],
                 json_schema: Dict[str, Any]) -> str:
    tax_str = "\n".join(
        f"- {t['id']}: {t.get('label','')} | {t.get('description','')}"
        for t in taxonomy_hits
    )

    ex_str = "\n".join(
        f"- title: {e.get('title','')}\n  doc_type: {e.get('doc_type','')}\n  topics: {e.get('topics',[])}\n  keywords: {e.get('keywords',[])}\n  abstract: {e.get('abstract','')}"
        for e in example_hits
    )

    schema_str = json.dumps(json_schema, indent=2)

    return f"""
You are generating STRUCTURED METADATA for document discovery and governance.

Rules:
- Output MUST be valid JSON only (no markdown).
- Use ONLY taxonomy IDs from the provided taxonomy options for controlled fields (topics/programs/geographies/sensitivity).
- If a field is unknown, use null or [] (do NOT guess).
- Keep abstract <= 80 words.
- Provide evidence_snippets: short phrases copied from the document that justify the key fields.

Document type hint: {doc_type}

DOCUMENT (high-signal excerpt):
{doc_text}

TAXONOMY OPTIONS (choose from these IDs):
{tax_str}

SIMILAR APPROVED EXAMPLES (for formatting guidance):
{ex_str}

JSON SCHEMA (must follow):
{schema_str}
""".strip()