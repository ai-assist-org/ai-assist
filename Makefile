.PHONY: help install install-dev test test-cov lint format clean pre-commit-install pre-commit-run sandbox-build

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package
	pip install -e .

install-dev:  ## Install package with dev dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=ai_assist --cov-report=html --cov-report=term

test-fast:  ## Run only fast tests (skip slow/integration)
	pytest -o 'addopts=-v --strict-markers --tb=short -n auto -m "not slow and not integration"'

lint:  ## Run linting (ruff)
	ruff check ai_assist tests

lint-fix:  ## Run linting with auto-fix
	ruff check --fix ai_assist tests

format:  ## Format code with black and isort
	black ai_assist tests
	isort ai_assist tests

format-check:  ## Check if code is formatted correctly (CI mode)
	black --check ai_assist tests
	isort --check ai_assist tests

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

pre-commit-install:  ## Install pre-commit hooks
	pre-commit install

pre-commit-run:  ## Run pre-commit on all files
	pre-commit run --all-files

pre-commit-update:  ## Update pre-commit hooks to latest versions
	pre-commit autoupdate

all-checks: format-check lint test  ## Run all quality checks (CI mode)

DCI_MCP_SERVER_DIR ?= ../dci-mcp-server

sandbox-build:  ## Build sandbox container images (ai-assist + dci-mcp-server)
	podman build -t ai-assist-sandbox -f sandbox/ai-assist/Dockerfile .
	podman build -t dci-mcp-server -f $(DCI_MCP_SERVER_DIR)/Containerfile.sse $(DCI_MCP_SERVER_DIR)

test-eval:  ## Run agent behavior eval suite (requires API key)
	uv run --extra eval python eval/run_eval.py --config eval/eval.yaml --model claude-sonnet-4-6 --no-llm-judges

test-eval-full:  ## Run eval suite with LLM judges (slower, requires claude CLI)
	uv run --extra eval python eval/run_eval.py --config eval/eval.yaml --model claude-sonnet-4-6

test-integration: sandbox-build-dev  ## Run sandbox integration tests (builds images first)
	pytest tests/test_sandbox_integration.py -v -m integration --override-ini="addopts="

sandbox-build-dev: sandbox-build  ## Build dev profile image (adds Go, Ansible, Python dev tools)
	podman build -t ai-assist-dev -f sandbox/profiles/dev/Dockerfile .

dev-setup: install-dev pre-commit-install  ## Complete dev environment setup
	@echo "✓ Development environment ready!"
	@echo "  Run 'make test' to verify everything works"
