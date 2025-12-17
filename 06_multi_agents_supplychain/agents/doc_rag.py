# agents/doc_rag.py
from __future__ import annotations
from typing import List, Tuple
from rag.retriever import Retriever
from utils.ollama_llm import generate

class DocRAGAgent:
    def __init__(self):
        self.retriever = Retriever()

    def answer(self, question: str, top_k: int = 4) -> str:
        hits = self.retriever.search(question, top_k=top_k)
        if not hits:
            return "I couldn't find anything in the docs."

        context_blocks: List[str] = []
        citations: List[str] = []
        for chunk, score in hits:
            context_blocks.append(f"[{chunk.doc_id}#{chunk.chunk_id}] ({chunk.source})\n{chunk.text}")
            citations.append(f"- {chunk.source} (chunk {chunk.chunk_id})")

        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""You are a helpful internal assistant. Answer ONLY using the provided context.
If the answer is not in the context, say you don't know.

Question:
{question}

Context:
{context}

Return:
1) A concise answer
2) A short "Sources" list with the chunk IDs
"""

        llm = generate(prompt)
        if llm:
            return llm.strip()

        # Fallback: extractive (no LLM)
        return (
            "LLM not available. Here are the most relevant passages:\n\n"
            + context
            + "\n\nSources:\n"
            + "\n".join(citations)
        )