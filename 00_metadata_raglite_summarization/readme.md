## Metadata RAG-lite summarization

Local, interview-friendly demo of **RAG-lite** for metadata generation:
- Retrieve **taxonomy options** (controlled vocabulary) + a few **similar approved examples**
- Use an LLM to generate **strict JSON metadata**
- Validate output against a schema + enforce taxonomy IDs

---

## CATER flow

### C — Collect

**Goal:** ingest documents and supporting context.

**Inputs**
- Documents to summarize: `data/docs/*.txt`
- Taxonomy (controlled vocab): `data/taxonomy/taxonomy.jsonl`
- Gold/approved examples: `data/examples/gold_metadata.jsonl`

**Code**
- Load files: `src/metaragl/io/loaders.py`

---

### A — Arrange and clean

**Goal:** normalize text and select the most informative parts of the document.

**Steps**
- Clean raw text (whitespace, newlines): `clean_text()`
- Extract “high-signal” excerpt (title/headers/first sections): `high_signal_slice()`
- (Optional heuristic) infer doc type: `simple_doc_type()`

**Code**
- Cleaning + excerpting: `src/metaragl/preprocess/clean.py`
- Doc type heuristic: `src/metaragl/preprocess/chunk.py`

---

### T — Train (prompt + retrieval + generation)

**Goal:** generate metadata using a constrained prompt + retrieval context.

This project does **not train** a model. Instead, “T” here means **LLM orchestration**:

**Retrieval (RAG-lite)**
- Build a small vector index over:
  - taxonomy records
  - approved example records
- Query the index using the document excerpt
- Retrieve:
  - Top K taxonomy hits (IDs + descriptions)
  - Top K example hits (approved metadata)

**Generation**
- Construct a prompt that includes:
  - document excerpt
  - retrieved taxonomy options
  - retrieved example records
  - output JSON schema
- Call the LLM backend:
  - `mock` (offline deterministic)
  - `ollama` (local model)

**Code**
- Hash embeddings (offline): `src/metaragl/retrieval/embedder.py`
- In-memory vector store: `src/metaragl/retrieval/vector_store.py`
- Index build + retrieval helpers: `src/metaragl/retrieval/retrieve.py`
- Prompt template: `src/metaragl/llm/prompts.py`
- LLM backends: `src/metaragl/llm/client.py`
- End-to-end orchestration: `src/metaragl/pipeline.py`

---

### E — Evaluate

**Goal:** ensure outputs are structurally correct and policy-compliant.

**Checks**
- Strict JSON parse (with a safe JSON extractor)
- Schema validation (required fields, types)
- Allowed-taxonomy enforcement (no made-up IDs)

**Code**
- Pydantic model + JSON schema: `src/metaragl/schema/metadata_schema.py`
- Validation + taxonomy enforcement: `src/metaragl/schema/validate.py`
- Safe JSON extraction + parse: `src/metaragl/pipeline.py` (`safe_json_loads()`)

---

### R — Release and run

**Goal:** run locally via CLI and iterate on taxonomy/examples.

**Run**
```bash
# install
python -m venv .venv
source .venv/bin/activate
pip install -e .

# choose backend
# (offline)
export METARAGL_LLM_BACKEND=mock

# (ollama)
# export METARAGL_LLM_BACKEND=ollama
# export OLLAMA_BASE_URL=http://127.0.0.1:11434
# export OLLAMA_MODEL=llama3.1:8b

# run
metaragl run data/docs/sample.txt
metaragl run data/docs/ag_prices.txt
metaragl run data/docs/public_release.txt
```

**Operational knobs (what you’d tune in prod)**
- `top_k_taxonomy`, `top_k_examples` (retrieval strength)
- taxonomy coverage (reduces “unknown” / null fields)
- prompt rules (e.g., sensitivity precedence rules)
- output validation strictness (block vs warn)

**Code**
- CLI entrypoint: `src/metaragl/cli.py`

---

## File map

### Data
- `data/docs/` — input documents for summarization
- `data/taxonomy/taxonomy.jsonl` — controlled vocabulary items
- `data/examples/gold_metadata.jsonl` — approved “gold” metadata examples

### Source
- `src/metaragl/cli.py` — CLI: `metaragl run <path>`
- `src/metaragl/pipeline.py` — end-to-end orchestration
- `src/metaragl/io/loaders.py` — JSONL + text file loading
- `src/metaragl/preprocess/clean.py` — text cleaning + high-signal extraction
- `src/metaragl/preprocess/chunk.py` — doc type heuristic
- `src/metaragl/retrieval/embedder.py` — offline hash embeddings
- `src/metaragl/retrieval/vector_store.py` — in-memory vector store
- `src/metaragl/retrieval/retrieve.py` — index build + retrieval
- `src/metaragl/llm/prompts.py` — prompt builder
- `src/metaragl/llm/client.py` — LLM backends (mock / ollama)
- `src/metaragl/schema/metadata_schema.py` — metadata model + JSON schema
- `src/metaragl/schema/validate.py` — schema + taxonomy validation

---

## Interview-ready “Was RAG used?” answer

**Yes — RAG-lite.** We used retrieval to ground generation with:
- a controlled taxonomy (allowed IDs + descriptions)
- a few similar approved examples

We did **not** build a full conversational RAG system over a large knowledge base; retrieval was used specifically to constrain metadata fields and improve consistency.
