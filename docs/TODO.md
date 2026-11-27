# To do

What the SDK still lacks next to a framework such as LangChain, to be
taken up after 0.1.0. Nothing here blocks the first release.

## Documentation

- [ ] A documentation site built from `docs/` (today: Markdown files in
      the repository).
- [ ] Examples by use case (a cookbook): a support desk with refunds, a
      procedure with a human approval, wrapping an existing agent loop,
      a local model.
- [ ] An API reference generated from the docstrings.
- [ ] A statement of what is stable: which names in `acuto.__all__` are
      covered by semantic versioning from 1.0, and how deprecations are
      announced.

## Long runs (in 0.1.0)

- [ ] Forking a checkpoint at an earlier step.
- [ ] Periodic snapshots, to bound replay time on very long runs.

## Later: building agent systems

- [ ] Several agents in one system, with contract-checked handoffs: only
      unit tests so far; needs runs on real tasks first. Not planned for
      0.2.0.
- [ ] Asynchronous and streaming interfaces (`async` model adapters,
      token streaming to the caller).

## Serving

- [ ] An agent served over HTTP (a small server around `acuto.Agent`), if
      users ask for it; today the SDK is a library only.
