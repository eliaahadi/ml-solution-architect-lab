# agents/router.py
from __future__ import annotations
import re
from typing import Dict, Any

def route(user_text: str) -> Dict[str, Any]:
    t = user_text.lower()

    # Policy intent
    if any(k in t for k in ["allowed", "permission", "policy", "compliance", "can i", "can we", "legal"]):
        return {"route": "POLICY", "facts": _extract_policy_facts(user_text)}

    # Data intent
    if any(k in t for k in ["inventory", "stock", "on hand", "shipment", "asn", "eta", "po", "purchase order", "order"]):
        return {"route": "DATA", "facts": {}}

    # Default to docs
    return {"route": "DOC_RAG", "facts": {}}

def _extract_policy_facts(user_text: str) -> Dict[str, Any]:
    t = user_text.lower()

    # SUPER naive fact extraction for MVP. In Gen-2 you’d gather facts via UI controls.
    role = "OPS"
    if "contractor" in t:
        role = "CONTRACTOR"
    if "procurement manager" in t:
        role = "PROCUREMENT_MANAGER"
    if "legal" in t:
        role = "LEGAL"

    action = "UNKNOWN"
    if any(k in t for k in ["supplier pricing", "pricing", "contract terms", "contracts"]):
        action = "VIEW_SUPPLIER_PRICING"
    if any(k in t for k in ["inventory", "on hand", "stock"]):
        action = "VIEW_INVENTORY"

    region = "US"
    m = re.search(r"\bregion\s*[:=]\s*([A-Za-z]{2})\b", user_text)
    if m:
        region = m.group(1).upper()

    return {"role": role, "action": action, "region": region}