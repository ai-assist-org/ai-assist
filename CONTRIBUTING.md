# Contributing to ai-assist

## Development Setup

### 1. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs:
- pytest (testing framework)
- pytest-cov (coverage; run `make test-cov` or `pytest --cov=ai_assist --cov-report=term`)
- pre-commit (git hooks)
- black (code formatting)
- ruff (linting and import sorting)

### 2. Set Up Pre-commit Hooks

```bash
pre-commit install
```

This installs git hooks that will automatically:
- Format code with black
- Lint and sort imports with ruff (ruff handles both; see `.pre-commit-config.yaml`)
- Fix trailing whitespace and line endings
- Run tests with pytest

### 3. Running Tests

```bash
# Unit tests (excludes integration tests by default)
make test                    # All non-integration tests (parallel)
make test-fast               # Skip slow + integration tests
make test-cov                # With coverage report (≥71% required)
pytest tests/test_file.py    # Specific test file
pytest -v -s                 # Verbose with print output

# Integration tests (require podman + built container images)
make test-integration        # Builds images then runs sandbox tests

# Eval tests (agent behavior validation, requires API key)
make test-eval               # Eval suite with check judges only
make test-eval-full          # Eval suite with LLM judges (slower, requires claude CLI)
uv run --extra eval python eval/run_eval.py --config eval/eval.yaml --model claude-sonnet-4-6 --cases case-name  # Single case
```

### 4. Code Quality

Run quality checks manually:

```bash
# Format code
black ai_assist tests

# Sort imports
ruff check --select I --fix ai_assist tests

# Lint code
ruff check ai_assist tests

# Fix auto-fixable issues
ruff check --fix ai_assist tests
```

Or run all checks at once:

```bash
pre-commit run --all-files
```

## Pre-commit Hooks

The pre-commit hooks will run automatically when you commit. They include:

### Formatting
- **black**: Formats Python code to a consistent style (120 char line length)
- **ruff**: Linting and import sorting (the `I` rule replaces isort; see `.pre-commit-config.yaml`)

### Linting
- **ruff**: Fast Python linter (checks for errors, style issues, bugs; also sorts imports)

### Basic Checks
- Remove trailing whitespace
- Fix end-of-file issues
- Check YAML/JSON/TOML syntax
- Prevent large files (>500KB)
- Check for merge conflicts
- Detect debug statements (pdb, etc.)

### Testing
- **pytest**: Runs all tests to ensure changes don't break functionality

## Commit Workflow

```bash
# Make your changes
vim ai_assist/some_file.py

# Stage changes
git add ai_assist/some_file.py

# Commit (pre-commit hooks run automatically)
git commit -m "Add new feature"

# If hooks fail, fix issues and try again
# The hooks will show you what needs to be fixed
```

## Skipping Hooks (Not Recommended)

If you absolutely need to skip hooks:

```bash
git commit --no-verify -m "Emergency fix"
```

⚠️ **Warning**: Only skip hooks for emergency fixes. Your PR may be rejected if code doesn't pass quality checks.

## Code Style Guidelines

### Line Length
- Maximum 120 characters per line
- Let black handle most formatting

### Imports
- Standard library imports first
- Third-party imports second
- Local imports last
- Sorted alphabetically within each group

### Type Hints
- Use type hints for function parameters and return types
- Example: `def read_file(path: str) -> str:`

### Docstrings
- Use docstrings for all public functions and classes
- Format: Google style or NumPy style

Example:
```python
def read_file(path: str, line_start: int = None) -> str:
    """Read a file from the filesystem.

    Args:
        path: Absolute path to the file
        line_start: Optional starting line number

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
    """
```

## Running Specific Test Suites

```bash
# Run only filesystem tool tests
pytest tests/test_filesystem_tools.py

# Run only agent tests
pytest tests/test_agent*.py

# Run only security tests
pytest tests/test_filesystem_security.py

# Run with coverage (target: ≥71%; build fails below 71%)
make test-cov

# Tests run in parallel by default (pytest-xdist, -n auto in pyproject.toml)
```

## Debugging Failed Tests

If tests fail during pre-commit:

```bash
# Run tests manually to see full output
pytest -v

# Run specific failing test
pytest tests/test_file.py::test_function -v

# Run with print statements visible
pytest -s
```

## Adding New Tests

When adding new functionality, add corresponding tests:

1. Create test file: `tests/test_your_feature.py`
2. Import pytest and your module:
   ```python
   import pytest
   from ai_assist.your_module import YourClass
   ```
3. Write test functions:
   ```python
   def test_your_function():
       result = your_function("test")
       assert result == "expected"
   ```
4. For async functions:
   ```python
   async def test_async_function():
       result = await async_function()
       assert result == "expected"
   ```

## Development Philosophy

### TDD — test first

1. **Red**: write a failing test that defines the behaviour.
2. **Green**: write the minimum code that makes it pass.
3. **Refactor**: clean up while keeping the suite green.

Baseline assumption: all tests are green before your change. If they fail
afterwards, your change broke them — fix it.

- Unit tests: a single function or class in isolation.
- Integration tests: MCP connections and tool execution.
- End-to-end tests: query → response, monitor cycles, AWL runs.
- Mock external services; cover error handling and graceful degradation.

### Snapshot / replay testing

The agent loop can be driven offline and deterministically by swapping the single
model seam (`AiAssistAgent.anthropic`) for a cassette-driven fake client
(`ai_assist/testing/`). A cassette is a JSON file listing the assistant turns
for one query; the Nth `messages.create`/`messages.stream` call returns the Nth
recorded turn (turn-index matching, robust to prompt drift). Only the LLM is
faked.

Unit and loop tests use the `make_replay_agent` fixture (`tests/conftest.py`):

```python
async def test_something(make_replay_agent):
    agent = await make_replay_agent([
        {"content": [{"type": "tool_use", "id": "t1", "name": "internal__think",
                      "input": {"thought": "plan"}}], "stop_reason": "tool_use"},
        {"content": [{"type": "text", "text": "Done"}], "stop_reason": "end_turn"},
    ])
    assert await agent.query(prompt="do it") == "Done"
```

Internal tools run for real here against an isolated temp KG, so only call safe
tools in hand-written cassettes. Examples: `tests/test_snapshot_replay.py`.

Eval cases use the same mechanism: `AI_ASSIST_EVAL_MODE=record` writes
`eval/dataset/cases/<case>/cassette.json`; `AI_ASSIST_EVAL_MODE=replay` runs
offline against it (`eval/eval_wrapper.py`). Eval replay does **not** execute
tools — it records tool selection for the judges and skips side effects.

### DRY

Extract logic used in more than one place. Config loading lives in
`config.py`; tool execution goes through `_execute_tool` in `agent.py`; query
patterns belong in a shared helper. If two features need the same code, move it
to one place instead of copying.

### Tracer bullets

Build end-to-end first, then enhance. Add the smallest path that proves the
architecture works (console output before a notifier, a static page before a
dashboard), keep the suite green, and iterate. Prefer three small working steps
over one large unfinished feature.

### Minimize branches

Fewer code paths, fewer bugs. Fail fast with a clear error instead of adding
fallbacks for missing dependencies or "just in case" failures. Handle missing
dependencies at install time (`pyproject.toml`). Branches are for user-facing
choices and data-driven cases.

### Portability (Linux and macOS)

Shipped shell scripts, the shell code ai-assist runs, and commands it tells a
model to run must work on both platforms: bash 3.2 (no `mapfile`, associative
arrays, `coproc`, `${var,,}`/`${var^^}`), POSIX awk (no `mktime`, `strftime`,
`gensub`, `> "/dev/stdout"`, backslashes in `gsub()` replacements), BSD userland
(no `sed -i`, `date -I`, `stat -c`, `readlink -f`, `grep -P`, `find -printf`),
and `sha1sum` only with a `shasum -a 1` fallback. `scripts/check_portability.py`
runs in pre-commit; `tests/test_check_portability.py` runs it over every tracked
script. New awk helpers are also exercised under `gawk --posix` by the `awk_dir`
fixture in `tests/test_self_improvement_workflow.py`.

## Continuous Integration

Pre-commit hooks run the same checks that will run in CI/CD, so:
- ✅ If pre-commit passes, CI will likely pass
- ❌ If pre-commit fails, CI will definitely fail

Fix issues locally before pushing!

## Questions?

If you have questions about:
- Pre-commit setup: Check `.pre-commit-config.yaml`
- Tool configuration: Check `pyproject.toml`
- Test configuration: Check `[tool.pytest.ini_options]` in `pyproject.toml`

## Quick Reference

```bash
# Install dev environment
pip install -e ".[dev]"

# Setup hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files

# Run tests
pytest

# Format code
black ai_assist tests

# Lint code
ruff check --fix ai_assist tests
```
