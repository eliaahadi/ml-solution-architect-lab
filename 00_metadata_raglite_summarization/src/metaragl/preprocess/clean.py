import re

def clean_text(text: str) -> str:
    # simple cleanup: normalize whitespace, drop repeated separators
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def high_signal_slice(text: str, max_chars: int = 4500) -> str:
    """
    For metadata summarization, you usually want title/subject/date lines + first sections.
    This is a cheap heuristic demo.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    head = "\n".join(lines[:80])
    return head[:max_chars]