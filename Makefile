.PHONY: all run quick data lint fmt test clean

# Full reproduction: pull data if needed, run all three methods with a
# bootstrap CI on the ML estimate, write tables and figures.
all: run

run:
	uv run python run.py --bootstrap 12

# Faster run without the bootstrap CI.
quick:
	uv run python run.py

# Pull and cache the raw data only.
data:
	uv run python -m frb.data

lint:
	uv run ruff check frb run.py tests
	uv run ruff format --check frb run.py tests

fmt:
	uv run ruff format frb run.py tests
	uv run ruff check --fix frb run.py tests

test:
	uv run pytest -q

clean:
	rm -f figures/*.png outputs/*.json outputs/*.md outputs/*_log.txt
