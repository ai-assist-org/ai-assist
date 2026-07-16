DATASET = locomo10.json
DATASET_URL = https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

.PHONY: download run run-quick run-baseline clean help

help:
	@echo "Usage:"
	@echo "  make download      - Download LoCoMo dataset"
	@echo "  make run           - Full benchmark with synthesis pipeline"
	@echo "  make run-quick     - Quick test (1 conversation, full pipeline)"
	@echo "  make run-baseline  - Vector-only baseline (no synthesis)"
	@echo "  make clean         - Remove dataset and temp files"

download: $(DATASET)

$(DATASET):
	curl -fSL -o $(DATASET) $(DATASET_URL)

run: $(DATASET)
	uv run python run_benchmark.py --dataset $(DATASET) --output results.md -v

run-quick: $(DATASET)
	uv run python run_benchmark.py --dataset $(DATASET) --limit 1 --output results-quick.md --keep-db -v

run-baseline: $(DATASET)
	uv run python run_benchmark.py --dataset $(DATASET) --skip-synthesis --output results-baseline.md -v

clean:
	rm -f $(DATASET) results.md results-quick.md results-baseline.md
	rm -rf /tmp/locomo-kg-*
