# Long runs: compaction, invariance, checkpoints

A long task outgrows the context window and outlives the process. Acuto
handles both with one design rule from [DESIGN.md](DESIGN.md) section 1: the agent state (the
messages) and the symbolic control state (monitors, facts, obligations)
are separate. Compaction rewrites only the first. Checkpoints persist the
second as a log of its inputs.

## 1. Compaction

A compactor (`acuto.loop.Compactor`) rewrites the message list when the
context outgrows its threshold. The threshold is measured from the
provider's reported context size, cached tokens included. The shipped
compactors form a ladder, from the cheapest and least lossy to the
costliest:

| Compactor | What it does |
|---|---|
| `MaskCompactor.admit` | Offloads a result over `max_result_tokens` when it arrives; the model sees its head, its tail and a recall id. |
| `MaskCompactor` | Replaces old tool results with recall stubs, in one batch, down to `target` x threshold. |
| `TruncateCompactor` | Drops turns older than the last `keep_turns`, keeps the user's own messages, and archives the dropped span. |
| `SummaryCompactor` | Replaces old turns with a structured summary. Between its two thresholds it runs only when no obligation is pending. |

What a compactor takes out goes to a `ContextArchive`. The model reads it
back with `acuto_recall`, so the view plus the archive is always the
whole history. Nothing is lost, only moved out of view.

These are never compacted:

- the task;
- the user's own messages;
- every assistant turn (the calls and their arguments);
- failed results;
- results under `min_result_tokens`;
- results of `pin_tools`.

The control state reaches the conversation as an appended note
(`control_state_text`), written when it changes and never deleted
mid-run. Deleting a note would rewrite the cached prompt prefix. Old
notes are dropped at the next compaction instead.

The design choices follow the evidence. Masking matches summarization at
half the cost ([arXiv:2508.21433](https://arxiv.org/abs/2508.21433)). Naive summaries lose the agent's place
([arXiv:2608.06503](https://arxiv.org/abs/2608.06503)). Truncation and summaries drop constraints stated
mid-session ([arXiv:2608.11242](https://arxiv.org/abs/2608.11242), [arXiv:2606.22528](https://arxiv.org/abs/2606.22528)).

## 2. Compaction invariance

**Setting.** A run feeds its controller a sequence of inputs, the
*journal* `J = j_1 ... j_n`. Each input is one of:

- an executed call with its result;
- an idle event;
- a call observed from outside the run;
- a fact written from outside.

The controller is a deterministic transition system:
`σ_k = δ(σ_{k-1}, j_k)`, where `σ` holds the monitors' residuals and
assumption flags, the fact store, the obligation ledger and the event
count (`SymbolicController.state_digest()` fingerprints it). Every
decision the controller makes is a function of `σ` and the flow state it
is given:

- the admissible set (masking);
- routing and fast-forward, including argument binding;
- `validate`, `verify_return` and `due_obligation`;
- the end-of-session verdict.

**Assumptions.**

- **A1.** The flow state given to the controller does not depend on the
  message view. `FreeFlow` passes the user's own state. `GraphFlow`
  passes its current node, whose transitions read facts.
- **A2.** Facts are written only by extractors over tool results and by
  explicit `assert_fact` calls. They are never read off model text.
  Predicate evaluators read facts and flow state only.

**Theorem (compaction invariance).** Under A1 and A2, take any two runs
whose journals agree on their first `k` entries, whatever compactors
they use and however often they recall. Their controllers are in the same
state after `k` entries, and every decision either controller makes at
that point is the same.

**Proof sketch.** By induction on `k`. A compactor receives the messages
and the archive, and the control state only as rendered text; it holds
no reference to the controller. `acuto_recall` is intercepted by the
loop before validation and never calls a journaled method, so it adds no
journal entry. Each journaled method is deterministic in `(σ, j)`:
event timestamps are the event count, and caches only memoize
deterministic probes. So equal journal prefixes give equal states. By A1
and A2, every decision is a function of the state and of inputs that are
themselves independent of the view.

**What follows.**

- **A rule in the contract library holds to the same standard at step
  1,000 as at step 1, under any compactor.** Every call the model
  proposes is validated against the same `σ` it would face with full
  context. An obligation triggered before a compaction is still pending
  after it, and still blocks the end of the session.
- **Compaction can change what the model does next.** It changes which
  calls are proposed, and so the trace, the task outcome and the cost.
  It cannot change what is allowed, what is required, or what is
  reported as a violation of a given trace.
- **The guarantee covers contracts, not prose.** A rule that exists only
  in the prompt or the conversation is outside it.
- **The assumptions are where the guarantee can break.** A custom flow
  whose state reads the messages breaks A1. A predicate evaluator that
  asks a model about the conversation breaks A2.

**Mechanical checks.**

- `tests/test_checkpoint.py::test_the_control_state_is_invariant_under_every_compactor`
  runs one fixed sequence of calls under no compactor, truncation,
  masking (with a recall turn) and summarization. It asserts equal state
  digests and equal journals.
- The resume tests assert that a replayed journal reproduces the digest.
- `state_digest` is stable across processes, so experiments can compare
  digests between runs.

## 3. Checkpoint and resume

A checkpoint is the journal, not a dump of the state. Resume replays the
journal into a fresh controller and recomputes `σ`, so no internal
structure is serialized.

- **One run:** `RunResult.checkpoint` holds the journal, the flow
  snapshot, the archive and the counters.
  `iterate/run/Agent.run(resume=...)` continues from it.
- **Crash safety:** with `checkpoint_path`, the checkpoint is written
  atomically before every step. Every call is recorded before it is sent
  (`in_flight`), so a step is the unit of atomicity. Resuming a step that
  was cut off after sending a call to a tool not marked `idempotent`
  raises `InFlightError`, unless the caller passes `on_in_flight="redo"`.
- **Version stamps:**
  - **Changed contracts are allowed.** The replay judges the recorded
    trace under the new contracts, and the change is traced as
    `checkpoint/contracts_changed`.
  - **Changed tool schemas are refused.** Recorded calls may no longer
    mean what they meant.

**Not yet done:**

- forking a checkpoint at an earlier step;
- periodic snapshots to bound replay time on very long runs.
