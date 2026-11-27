# Contributing to Acuto

Thanks for your interest. This page covers the development setup, the
checks a change must pass, and where different kinds of changes belong.

## Development setup

Acuto depends on [ContrAgent](https://github.com/SponsioLabs/ContrAgent),
the contract kernel, which the install below takes from PyPI:

```bash
pip install -e ".[dev]"
pre-commit install          # optional: runs ruff on every commit
```

To work against a ContrAgent checkout instead, install that first with
`pip install -e ../ContrAgent`.

Python 3.10 or newer. The framework itself has no third-party
dependencies beyond ContrAgent; model providers are reached over plain
HTTP from `acuto/models/`.

## Before you open a pull request

```bash
ruff check .
ruff format --check .
pytest -q
```

All three run in CI on Python 3.10 through 3.13. No test needs an API
key. Model behavior is scripted with `acuto.models.MockModel`.

A change to the runtime (`acuto/`) needs a test. The suite is organized
by mechanism: `test_masking.py`, `test_jump_forward.py`,
`test_completion_check.py`, `test_ag.py` (assume-guarantee semantics),
`test_loop.py` (the built-in loop end to end), `test_controller.py`
(the embedding facade), `test_api.py` (the user-facing bundle). Put the
new test next to the mechanism it exercises.

## Where things go

| You want to | Change |
|---|---|
| Add or change a contract-authoring form | `acuto/contracts.py` (builder + compiler), a test in `tests/test_masking.py` or `test_jump_forward.py`, and the DSL table in `docs/SDK.md` |
| Change what the controller decides | `acuto/control.py`; keep `admissible_actions` cheap and `validate` complete |
| Change the step protocol (mask → route → verify → commit) | `acuto/controller.py` first; `acuto/loop.py` consumes it |
| Add a model provider | `acuto/models/<provider>.py` implementing `generate(messages, tools, system) -> ModelTurn`, registered in `acuto/models/__init__.py` |
| Add a flow kind | implement the `Flow` protocol in `acuto/flow.py` |
| Add a runnable example | `examples/`, scripted model by default so it runs without credentials |

Public names are exported from `acuto/__init__.py` in three tiers
(write an agent / embed in a loop / extend). Adding a name there means
documenting it in `docs/SDK.md`.

## Style

`ruff` enforces lint and formatting in CI (line length 100). Prefer plain,
direct docstrings that say what a function decides and what it returns;
the controller never answers with a bare boolean, and new decision
types should follow `acuto/decisions.py`.

## Reporting a bug

Open an issue with the contract library (or the smallest subset that
reproduces it), the tool names, the trace (`trace_path=` writes JSONL),
and what you expected the controller to decide. A failing test is the
best report.

## Releasing

Releases are built and published by `.github/workflows/release.yml`, in
the public repository only, with trusted publishing (no token is stored).

1. Set the version in `pyproject.toml` and `acuto/__init__.py`, and date
   the changelog entry.
2. Tag the commit `vX.Y.Z` and push the tag. The workflow checks that the
   tag matches the version, builds the files once, checks them, and
   uploads them to TestPyPI.
3. Install dependencies from PyPI, then install the wheel from TestPyPI.

   ```bash
   pip install 'contragent>=1.2,<2'
   pip install --index-url https://test.pypi.org/simple/ --no-deps --only-binary=:all: acuto==X.Y.Z
   ```

4. Publish the same files to PyPI:
   `gh workflow run release.yml -f publish_run=<id of the run of step 2>`

A published version cannot be replaced; a mistake is fixed by a new
version.
