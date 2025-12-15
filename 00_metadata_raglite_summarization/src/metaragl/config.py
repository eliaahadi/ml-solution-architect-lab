from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    llm_backend: str = os.getenv("METARAGL_LLM_BACKEND", "mock")  # mock | ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")

    # retrieval
    embed_dim: int = 512
    top_k_taxonomy: int = 12
    top_k_examples: int = 3

settings = Settings()