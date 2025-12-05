.PHONY: train serve test mlflow-ui

train:
	cd 01_tabular-ml-mlops && python -m src.train

serve:
	cd 01_tabular-ml-mlops && uvicorn src.api:app --reload

test:
	cd 01_tabular-ml-mlops && PYTHONPATH=. pytest -q

mlflow-ui:
	cd 01_tabular-ml-mlops && mlflow ui