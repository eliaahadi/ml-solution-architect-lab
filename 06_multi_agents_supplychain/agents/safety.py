# agents/safety.py
from __future__ import annotations

class SafetyAgent:
    def postprocess(self, route: str, response: str) -> str:
        # Minimal: if doc route, enforce presence of Sources section
        if route == "DOC_RAG" and ("Sources" not in response and "SOURCE" not in response):
            return response + "\n\nSources:\n- (missing) Please rebuild answer with explicit citations."
        return response