# ML Solution Architect Lab

This repo is my personal lab to practice end to end machine learning, deep learning, GenAI, and reinforcement learning projects the way a solution architect would design them.

Each project focuses on:
- Clear problem framing
- Data ingestion and preprocessing
- Model training and evaluation
- Deployment or API surface
- Architecture and tradeoffs

Everything is built using open source tools or free tiers that run locally on a Mac.

---

## Projects

| # | Folder                         | Topic                        | ML type                      | Highlights                                                      |
|---|--------------------------------|------------------------------|------------------------------|-----------------------------------------------------------------|
| 1 | `01_tabular-ml-mlops`         | Credit risk or housing ML    | Classic ML                   | scikit learn, MLflow, FastAPI, REST inference, tests           |
| 2 | `02_cv-deep-learning`         | Image classification         | Deep learning (CV)           | PyTorch/TensorFlow, image pipeline, experiment tracking        |
| 3 | `03_nlp-genai-rag`            | RAG over local documents     | GenAI / LLM                  | local LLM, vector store, RAG API                               |
| 4 | `04_time-series`              | Demand or traffic forecasting| Time series                  | classical + ML forecasting, forecasting API                    |
| 5 | `05_reinforcement-learning`   | CartPole control             | Reinforcement learning       | gymnasium, stable baselines, policy training and evaluation    |

There is also a `shared/` folder for utilities, configs, and Docker files that are reused across projects.

---

## Tech stack

Core tools:

- Python 3.11+ (local virtualenv)
- `pandas`, `numpy`
- `scikit-learn`
- `PyTorch` or `TensorFlow` (for deep learning)
- `MLflow` for experiment tracking
- `FastAPI` + `uvicorn` for serving models
- `pytest` for tests
- `gymnasium`, `stable-baselines3` for RL
- `sentence-transformers`, `Chroma` or `FAISS` for GenAI RAG

All projects are designed to run locally on CPU, with optional GPU acceleration if available.

---

## Repository layout

```text
ml-solution-architect-lab/
  README.md
  requirements.txt            # or pyproject.toml
  .gitignore

  01_tabular-ml-mlops/
    notebooks/
      01_eda.ipynb
      02_baseline_model.ipynb
    src/
      __init__.py
      config.py
      data.py
      features.py
      model.py
      train.py
      api.py
    tests/
      test_data.py
      test_inference.py
    mlruns/                   # MLflow tracking (local)
    README.md

  02_cv-deep-learning/
    notebooks/
    src/
    models/
    README.md

  03_nlp-genai-rag/
    notebooks/
    src/
    api/
    README.md

  04_time-series/
    notebooks/
    src/
    README.md

  05_reinforcement-learning/
    notebooks/
    src/
    README.md

  shared/
    utils/
    docker/
```

## Quick Start
```
python3 -m venv .venv

source .venv/bin/activate   # on macOS / Linux

.venv\Scripts\activate    # on Windows

pip install --upgrade pip

pip install -r requirements.txt
```