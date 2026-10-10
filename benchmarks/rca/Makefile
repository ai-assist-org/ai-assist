JOB_ID ?= d47dd2ec-253e-4fe6-838c-952966b2bdff
MODELS ?=

.PHONY: run clean help

help:
	@echo "Usage:"
	@echo "  make run JOB_ID=<id> MODELS=name1,name2  - Run the comparison (MODELS empty = all)"
	@echo "  make clean                                - Remove downloaded report/meta files"

run:
	uv run python run_comparison.py --job-id $(JOB_ID) $(if $(MODELS),--models $(MODELS),) -v

clean:
	rm -rf results/*/
