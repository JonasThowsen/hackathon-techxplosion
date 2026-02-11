default: check

# Run all checks
check: lint typecheck

# Lint and auto-fix
lint:
    uv run ruff check . --fix
    uv run ruff format .

# Lint without fixing (for CI)
lint-ci:
    uv run ruff check .
    uv run ruff format . --check

# Type checking
typecheck:
    uv run basedpyright .

# Full CI pipeline
ci: lint-ci typecheck
