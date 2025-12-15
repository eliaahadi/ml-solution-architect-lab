import math
import re
from typing import List

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

class HashEmbedder:
    """
    Fully offline, deterministic embedding using a hashing trick.
    Not state-of-the-art, but perfect for a local demo of RAG-lite.
    """
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _tokenize(self, text: str) -> List[str]:
        return _TOKEN_RE.findall(text.lower())

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        for tok in tokens:
            h = hash(tok)  # deterministic within run; good enough for demo
            idx = abs(h) % self.dim
            sign = -1.0 if (h & 1) else 1.0
            vec[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(v*v for v in vec)) or 1.0
        return [v / norm for v in vec]