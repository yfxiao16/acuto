# Changelog

All notable changes to Acuto are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [semantic versioning](https://semver.org/).

## [0.1.1]

Initial SDK release. The framework includes the contract DSL (`action` /
`after` / `when` / `ag`), `ContractLibrary` with symbolic verification,
`SymbolicController` (masking, fast-forward, verification, obligations,
residual policy cache), `Agent` and the runtime loop, `FreeFlow` /
`GraphFlow`, model adapters (OpenAI-compatible, Anthropic, Gemini,
Bedrock, ReAct wrapper, Mock), and long-run support with context
compaction, recall, checkpoints, and resume.

### Added
- `acuto.Controller`: one session of contract-based control that any
  agent loop can drive (`mask`, `route`, `verify`, `commit`, `execute`,
  `due_obligation`, `assert_fact`, `finalize`), with `CallOutcome` and
  `as_action`. The built-in loop now runs on it.
- Return-event verification: a tool result that would falsify a
  contract's assumption is withheld from the model and writes no state
  (`SymbolicController.verify_return`, `ReturnVerdict`).
- `verbose=True` on `Agent.run` / `Agent.stream` / `acuto.run`, and
  `acuto.console.render(result)`: a terminal rendering of every
  decision (masked, proposed, fast-forwarded, refused, withheld) with a
  banner and a summary.
- `acuto.models.resolve` reads a `.env` in the working directory when
  the provider's key is not in the environment.
- `examples/quickstart.py`, `examples/embed_controller.py`, `docs/SDK.md`, `CONTRIBUTING.md`,
  CI on Python 3.10 through 3.13 with `ruff format` enforced, `py.typed`.
- Long runs (`docs/LONG_RUNS.md`): compactors (`MaskCompactor`,
  `TruncateCompactor`, `SummaryCompactor`) with a `ContextArchive` the
  model reads back through `acuto_recall`; `Checkpoint`, written before
  every step with `checkpoint_path=` and continued with `resume=`, on
  `acuto.run`, `acuto.iterate` and `Agent.run` / `Agent.stream`; a call
  that was in flight when a run died is not repeated without
  `on_in_flight="redo"` unless its tool is marked `idempotent`
  (`InFlightError`). `examples/long_run.py`.
- Per-value contracts (ContrAgent's `ForEach`) and argument terms judged
  at validation.
- Bedrock: a throttled request moves to another region
  (`ACUTO_BEDROCK_FALLBACK_REGIONS`) and can wait for the quota
  (`ACUTO_BEDROCK_QUOTA_WAIT`); prompt caching for Anthropic models.
- `examples/custom_model.py` and a README section on using your own model:
  the shipped providers, any OpenAI-compatible endpoint, and a model behind
  your own API.

### Changed
- Runtime settings use `ACUTO_<NAME>`, such as `ACUTO_MODEL_MAX_RETRIES`
  and `ACUTO_BEDROCK_TIMEOUT`.
- ContrAgent is a declared dependency (`contragent>=1.2,<2`, from
  PyPI); no separate clone is needed. The package imports from
  ContrAgent's public modules only (`contragent`, `contragent.formulas`,
  `contragent.kernel`).

### Fixed
- Custom OpenAI-compatible endpoints use credentials only when explicitly
  configured.
- The tool registry rejects async functions, and an awaitable result is
  reported as a tool error.
- A fast-forward prediction that verification blocks is withheld and the
  model decides the step; the session used to end as a dead end.
- A guarantee `F fact(p)` met by the run's last call is no longer listed
  as broken in the final report.
- A text turn is held back while an obligation due next is pending, not
  only one due eventually. A call that ran and returned an error can
  count as called (`after(...).asserts(..., on_error=True)`,
  `when(...).obligates(..., discharged_on_error=True)`; both off by
  default).
- Bedrock: request signatures take their date from the real-time clock,
  so a simulator that freezes the process's clock no longer has every
  request rejected as expired; the request timeout is a setting (180
  seconds by default).

[0.1.1]: https://github.com/yfxiao16/acuto/releases/tag/v0.1.1
