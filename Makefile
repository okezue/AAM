.PHONY: install test lint smoke synthetic zip

install:
	python -m pip install -e .[dev]

test:
	pytest

lint:
	ruff check src tests scripts

smoke:
	aam smoke --config configs/experiments/l0_cpu_smoke.yaml

synthetic:
	aam run configs/experiments/l0_cpu_smoke.yaml

zip:
	cd .. && zip -qr activation-associative-memory.zip activation-associative-memory
