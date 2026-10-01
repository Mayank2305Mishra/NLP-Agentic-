.PHONY: validate build-index query benchmark test ui health

validate:
	python -m scripts.validate_dataset

build-index:
	python -m scripts.build_index

query:
	python -m scripts.query --help

benchmark:
	python -m scripts.benchmark --help

health:
	python -m scripts.query --health

test:
	python -m pytest -q

test-live:
	RUN_LIVE_OLLAMA_TESTS=1 python -m pytest -q -m live

ui:
	streamlit run app/ui.py --server.address 127.0.0.1
