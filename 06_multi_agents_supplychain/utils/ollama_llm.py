# utils/ollama_llm.py
from __future__ import annotations
import requests
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/generate"

def ollama_available(timeout_s: float = 0.3) -> bool:
    try:
        r = requests.get("http://localhost:11434", timeout=timeout_s)
        return r.status_code < 500
    except Exception:
        return False

def generate(prompt: str, model: str = "llama3.1:8b", temperature: float = 0.2) -> Optional[str]:
    """
    Returns generated text if Ollama is running, else None.
    """
    if not ollama_available():
        return None

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response")
    except Exception:
        return None