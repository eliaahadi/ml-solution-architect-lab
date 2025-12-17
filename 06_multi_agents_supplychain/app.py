# app.py
from __future__ import annotations

 
from agents.doc_rag import DocRAGAgent
from agents.data_sql import DataAgent
from agents.policy import PolicyAgent
from agents.safety import SafetyAgent


import re
from typing import Any, Dict


def route(user_text: str) -> Dict[str, Any]:
    """Simple deterministic router.

    Returns:
      {"route": "POLICY"|"DATA"|"DOC_RAG", "facts": {...}}
    """
    t = user_text.lower().strip()

    # Policy intent
    if any(k in t for k in ["allowed", "permission", "policy", "compliance", "can i", "can we", "legal"]):
        return {"route": "POLICY", "facts": _extract_policy_facts(user_text)}

    # Data intent
    if any(k in t for k in [
        "inventory", "stock", "on hand",
        "shipment", "asn", "eta", "delay",
        "po", "purchase order", "order",
    ]):
        return {"route": "DATA", "facts": {}}

    # Default to docs
    return {"route": "DOC_RAG", "facts": {}}


def _extract_policy_facts(user_text: str) -> Dict[str, Any]:
    """Very small, MVP-grade fact extractor.

    In Gen-2 you typically collect these as structured UI fields (role/action/region)
    rather than parsing free text.
    """
    t = user_text.lower()

    # Role
    role = "OPS"
    if "contractor" in t:
        role = "CONTRACTOR"
    if "procurement manager" in t:
        role = "PROCUREMENT_MANAGER"
    if "legal" in t:
        role = "LEGAL"

    # Action
    action = "UNKNOWN"
    if any(k in t for k in ["supplier pricing", "pricing", "contract terms", "contracts"]):
        action = "VIEW_SUPPLIER_PRICING"
    if any(k in t for k in ["inventory", "on hand", "stock"]):
        action = "VIEW_INVENTORY"

    # Region (optional, default US). Accepts patterns like "Region=IR" or "region: IR"
    region = "US"
    m = re.search(r"\bregion\s*[:=]\s*([A-Za-z]{2})\b", user_text)
    if m:
        region = m.group(1).upper()

    return {"role": role, "action": action, "region": region}

def main():
    doc_agent = DocRAGAgent()
    data_agent = DataAgent()
    policy_agent = PolicyAgent()
    safety = SafetyAgent()

    print("Local multi-agent chat. Type 'exit' to quit.\n")

    while True:
        user = input("You> ").strip()
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        r = route(user)
        which = r["route"]

        if which == "POLICY":
            facts = r["facts"]
            decision = policy_agent.decide(facts)
            resp = (
                f"Policy decision: **{decision.effect}**\n"
                f"- Reason: {decision.reason}\n"
                f"- Matched rule: {decision.rule_id}\n"
                f"- Facts: {facts}\n"
                f"- Citations:\n" + "\n".join([f"  - {c}" for c in decision.citations])
            )
        elif which == "DATA":
            resp = data_agent.query(user)
        else:
            resp = doc_agent.answer(user)

        resp = safety.postprocess(which, resp)
        print(f"\nAssistant ({which})>\n{resp}\n")

if __name__ == "__main__":
    main()