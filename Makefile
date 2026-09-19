.PHONY: setup check test lint doctor smoke candidates protocol ml-smoke training-host-check

export PYTHONPATH := $(CURDIR)/src

setup:
	uv sync --extra dev

test:
	uv run --no-sync pytest

lint:
	uv run --no-sync ruff check .

doctor:
	uv run --no-sync python -m go2wm doctor

smoke:
	uv run --no-sync python -m go2wm fake-smoke

candidates:
	uv run --no-sync python -m go2wm candidates

protocol:
	uv run --no-sync python scripts/verify_protocol.py

training-host-check:
	uv run --no-sync python scripts/check_go2_training_host.py

check: lint test doctor smoke candidates protocol

ml-smoke:
	uv run --no-sync python -m go2wm.learning smoke --out runs/ml-smoke-$$(date -u +%Y%m%dT%H%M%SZ)
