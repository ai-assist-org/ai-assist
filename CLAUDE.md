# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT: Read `AGENTS.md` before writing any code.** It contains the project architecture, development guidelines, coding patterns, and examples that must be followed.

## Development Commands

### Setup
```bash
# Install with dev dependencies
uv sync
# or: pip install -e ".[dev]"

# Setup pre-commit hooks
make pre-commit-install
# or: pre-commit install
```

### Testing
```bash
make test                    # Run all tests (pytest with -n auto for parallelism)
make test-cov                # Run tests with coverage report (≥71% required)
make test-fast               # Skip slow/integration tests
pytest tests/test_file.py    # Run specific test file
pytest -v -s                 # Verbose with print output
```

### Code Quality
```bash
make format                  # Format with black and isort
make lint                    # Lint with ruff
make lint-fix                # Auto-fix linting issues
make pre-commit-run          # Run all pre-commit hooks
make all-checks              # format-check + lint + test (CI mode)
```

### Running the Application
```bash
uv run ai-assist                      # Interactive mode (default)
uv run ai-assist /monitor             # Start monitoring mode
uv run ai-assist /query "..."         # One-off query
uv run ai-assist /run workflow.awl    # Execute AWL workflow
uv run ai-assist --dev /interactive   # Enable code auto-reload (dev mode)
```

### Service Management (Persistent Background Service)
```bash
ai-assist /service install           # Install as persistent service (default: ~/.ai-assist)
ai-assist /service install ~/.iris   # Install additional instance
ai-assist /service start|stop|restart|status <config_dir>
ai-assist /service enable|disable <config_dir>
ai-assist /service logs <config_dir> [-f]  # View logs (-f to follow)
ai-assist /service remove <config_dir>
```

## Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:
- black (formatting)
- ruff (linting and import sorting)
- mypy (type checking)
- bandit (security linting)
- pylint (duplicate code detection)
- pytest (test suite with coverage ≥71%)
- YAML/JSON/TOML syntax checks
- Trailing whitespace removal

Run `pre-commit run --all-files` before declaring a feature complete.

## Debugging
- Enable verbose logging: Set `logging.basicConfig(level=logging.DEBUG)` in code
- Test components individually: `python -c "from ai_assist.module import ..."`
- Check state: Inspect `~/.ai-assist/` directory contents
- Monitor mode logs: Check console output or use systemd logs if running as service
- Use `--dev` flag for code auto-reload during development

## Important Notes

- **All tests must be green before any change** (TDD baseline assumption)
- **No automatic truncation** - tools support `__save_to_file` to handle large results
- **Security**: Script execution is disabled by default. Only enable if you trust installed skills.
- **Extended context**: 1M context window is native for Claude 4.6+ models (no opt-in needed)
- **JSON processing**: `__jq_filter` parameter on any tool call for inline filtering; `internal__json_query` for processing saved files. Both require `jq` installed
- **Vertex AI users**: For Claude 4.6+ models, use dateless IDs (e.g., `claude-sonnet-4-6`). The `@YYYYMMDD` format only applies to older models (e.g., `claude-sonnet-4-5@20250929`)
