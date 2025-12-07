.PHONY: train serve test mlflow-ui

train:
	python -m src.train

serve:
	uvicorn src.api:app --reload

test:
	PYTHONPATH=. pytest -q

mlflow-ui:
	mlflow ui