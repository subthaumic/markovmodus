# Contributing to markovmodus

Thanks for your interest in improving `markovmodus`! Even small fixes help.

## Getting started

1. **Fork & clone** this repository.
2. Create a virtual environment and install the developer dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Install the pre-commit hooks so linting/tests/docs run automatically:
   ```bash
   pre-commit install
   pre-commit run --all-files
   ```
   Hooks rely on the bundled `.conda` environment (`./.conda/bin/...`), so ensure it is available locally.
3. Run the test suite to ensure everything is green before you begin:
   ```bash
   PYTHONPATH=src pytest
   ```

## Branching & commits

- Base all changes on the latest `main`.
- Use descriptive branch names, e.g. `feat/transition-api` or `fix/adata-metadata`.
- Squash or rebase before opening a pull request so the history stays tidy.
- Keep commits focused; include a concise message describing “what” and “why”.

## Coding guidelines

- Follow the existing code style; run `ruff` or `mypy` if you touch typed logic.
- Add or update tests alongside your change. New features should have coverage.
- Document public APIs in docstrings; keep README/docs snippets in sync.

## Pull requests

1. Ensure `pytest` passes locally (include coverage if practical).
2. Update documentation/changelog entries when behaviour changes.
3. Open a PR against `main`, summarising the change and its motivation.
4. Respond to review feedback; CI must be green before merge.

## Releasing

Release tags (`vX.Y.Z`) are made from `main` after the checklist in `README.md`
has been followed. Do not tag releases from feature branches.

## Questions?

Open a GitHub issue or start a discussion. Happy hacking!
