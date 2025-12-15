def simple_doc_type(text: str) -> str:
    t = text.lower()
    if "memo" in t or "subject:" in t:
        return "memo"
    if "abstract" in t or "introduction" in t:
        return "report"
    return "unknown"