# Local multi-agent RAG (free) with deterministic policy enforcement

This repo is a simple, local/free multi-agent system:
- Router agent: chooses doc RAG vs data vs policy
- Doc RAG agent: answers from ingested docs with citations
- Data agent: queries local SQLite supply chain tables (inventory/shipments/orders)
- Policy agent: deterministic allow/deny/escalate from YAML decision table
- Safety agent: lightweight guardrails (ex: enforce citations for doc-based answers)

## Quick start

### 1) Create venv + install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Create local DB
```bash
python data/init_db.py
```

### 3) Build local vector index from docs/
```bash
# either works; the -m form is more reliable
python -m rag.build_index
# or
python rag/build_index.py
```

### 4) Run chat
```bash
python app.py
```

```
rag-multi-agents-local/
  readme.md
  requirements.txt
  app.py
  config/
    policies.yaml
  data/
    init_db.py
    supplychain.db            # created at runtime
  docs/
    sample_sop.md
    supplier_terms.md
  rag/
    ingest.py
    build_index.py
    retriever.py
  agents/
    router.py
    doc_rag.py
    data_sql.py
    policy.py
    safety.py
  utils/
    ollama_llm.py
```

## Troubleshooting

### faiss-cpu install issues (macOS / Python versions)
If `pip install -r requirements.txt` fails for `faiss-cpu`, it’s usually because there isn’t a wheel for your Python version/platform.

Options:
1) Use Python 3.11 (recommended). Create a new venv with Python 3.11 and re-install.
2) If you use conda/mamba, install FAISS via conda instead:
   ```bash
   conda install -c conda-forge faiss-cpu
   ```
   Then remove/comment out `faiss-cpu` in `requirements.txt` for that environment.
3) If you want to stay pure-pip, swap FAISS for a simpler local index (e.g., `hnswlib`) and update `rag/build_index.py` + `rag/retriever.py`.