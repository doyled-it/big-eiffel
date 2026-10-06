.PHONY: all run quick data lint fmt test clean

# Full reproduction: pull data if needed, run all three methods, write tables
# and figures. The ML models are seeded, so this is deterministic.
all: run

run:
	uv run python run.py

# Add the optional (biased, diagnostic-only) bootstrap CI on the ML estimate.
boot:
	uv run python run.py --bootstrap 12

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

update:
	uv run python -m scripts.update_dataset --days 8

publish:
	uv run python scripts/publish_hf.py
