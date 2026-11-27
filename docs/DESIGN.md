# Acuto v0 Design

> This document records the v0 architecture. For the current SDK interface,
> see the [README](../README.md) and [SDK reference](SDK.md).

Acuto is a neuro-symbolic agent harness: the LLM decides *how* to accomplish
a task when genuine choices remain; a symbolic controller determines *what*
is admissible and directly executes actions when the procedure leaves no
choice. This document records the v0 architecture, what is reused from
ContrAgent, the extension seams deliberately preserved, and what is
deliberately deferred.

## 1. Two kinds of state

- **Agent state**: owned by the agent/application (messages, working
  memory, workflow node, arbitrary user data). Acuto never formalizes it;
  the controller receives it only as a read-only view (`agent_state`
  parameter), which is the extension point for future predicates whose
  truth depends on application state.
- **Symbolic control state**: owned by the `SymbolicController`, holding LTLf
  residuals (one DFA monitor per compiled rule), the `FactStore`, the
  `ObligationEngine`, and the grounding accumulators. It answers "what is
  the agent allowed or required to do", never "what is the agent doing".

The two interact at runtime but are never merged into a product FSM.

## 2. What is reused from ContrAgent (and what is not)

Acuto depends on `contragent` as a library and uses only its clean,
framework-free layers:

| ContrAgent component | Role in Acuto |
|---|---|
| `formulas/formula.py` (LTLf AST) | target of contract compilation |
| `formulas/dfa_evaluator.DFAEvaluator` | the contract backend: 3-valued residual monitor, O(1)/event, `snapshot()/restore()/peek()` make per-step admissibility probing cheap |
| `tracer/grounding.py` (`GroundingState`, `ground_event`) | event → proposition valuations (trace-level atoms: `called`, `count`, `arg_value`, ...) |
| `models/trace.Event` | canonical event record |
| `patterns/`, `generation/dsl_to_contract.py` | available through `acuto.CustomRule` for corpus-induced shapes |
| `formulas/sat.py` | future: dead-end/realizability lookahead (deferred) |

Deliberately **not** used: `BaseGuard` / `RuntimeMonitor`. They are a
check-only enforcement layer for adapting other frameworks (block + rollback
+ reprompt), hardcode the recursive verifier backend, and their
`filter_tools` probe is O(candidates × trace length) with a full re-ground
per probe. Acuto's controller is built directly on `DFAEvaluator`.

Because Acuto blocks violating actions *before* they commit, the live
monitors only ever consume compliant events; residuals never collapse to ⊥
in enforce flow, so none of ContrAgent's staleness/freshness machinery is
needed.

Upstream candidates for ContrAgent (small, later): `GroundingState.clone()`
(we deepcopy per probe today) and a DFA-backed `probe()` fast path on
`TraceVerifier` (its docstring already tracks this as a follow-up).

## 3. Pipeline

```
environment / agent state
        ↓  PredicateEvaluator        (facts → propositions)
        ↓  grounding(event)          (trace → propositions)
   symbolic propositions
        ↓  contract progression     (LTLfBackend: residual DFAs)
     control state
        ↓  action-space compilation
   admissible actions
```

The DFA reasons over propositions only; it never knows how a proposition
was established. `fact(p)` atoms are injected into each valuation by the
controller via the `PredicateEvaluator`; trace atoms come from ContrAgent
grounding.

## 4. Contract representation

Authoring surface (`acuto/contracts.py`), compiled into two halves:

| Concept | Author as | Compiles to |
|---|---|---|
| REQUIRES | `action(t).requires(*facts)` | LTLf `G(called(t) → ∧ fact(p))` |
| FORBIDS | `action(t).forbidden_when(*facts)` | LTLf `G(called(t) → ¬fact(p))` |
| ordering | `action(t).requires_before(*tools)` | LTLf `(¬called(t) U ∨called(p)) ∨ G(¬called(t))` |
| rate limit | `action(t).at_most(n)` | LTLf `G(called(t) → count(t) ≤ n)` |
| EVIDENCE_FOR | `after(t).asserts(fact, when=, value=, metadata=)` | controller metadata (result extractor) |
| INVALIDATES | `after(t).invalidates(*facts)` | controller metadata |
| OBLIGATES | `when(fact).obligates(tool, args=/binder=)` | controller metadata (`ObligationSpec`) |
| escape hatch | `CustomRule(formula)` | any raw ContrAgent LTLf formula |
| A/G contract | `ag(assumption, guarantee)` | two LTLf halves, see below |

**What an assumption means.** Acuto follows ContrAgent's reading, which
changed in September 2026. The guarantee is what the *agent* must keep and
it is in force from the first event of the session. The assumption is what
the *environment* is required to keep, so it is written over environment
predicates (`fact(...)`) and never over `called` / `count` / `arg_*`, which
the agent controls — `compile_contracts` rejects the latter, and rejects an
assumption demanding an unbounded eventuality, since suppression can
withhold an event but cannot bring one about. Weak until,
`(phi U psi) | G phi`, is a safety property and is allowed.

A contract falls IDLE, and stops binding the agent, only if an environment
event falsifies its assumption. Acuto does not wait for that. The return
event of every executed call goes through `SymbolicController.verify_return`
before the model sees the result: all facts the result would write are
applied on a checkpoint and checked jointly against every assumption. A
return that would falsify one is *suppressed* as a whole (tracing
`return/suppressed`): `update` writes none of its facts, and the loop hands
the model a withheld-result error in place of the result. Direct writes
through `assert_fact` keep a per-write probe (tracing `fact/suppressed`).
This is the agent-side `block` and environment-side `suppress` pair of
bidirectional runtime enforcement.

To scope a guarantee to a trigger the agent itself produces, put the
trigger inside the guarantee, as `G(trigger -> ...)` — the shape every
fluent builder in the table above already compiles to. The predecessor
mechanisms for this, `activate_at="first_match"` and assumption-as-trigger,
are gone.

**Why obligations are metadata, not LTLf.** A monitor decides an
undischarged `X(called(a))` only when the session ends: ContrAgent reads
what is still owed then on idle events after the last one, at which no tool
is called, so it fails. That is the right verdict for a *checker*, but it
comes after the session, too late to get the call made. Acuto is *active*:
the ObligationEngine tracks the obligation, the controller executes it
(jump-forward) when its arguments are procedurally determined, hands the
LLM a singleton action space when they are not, holds back a text turn
(which may end the session) while any obligation is pending, and
`finalize()` reports any still-pending obligation as a violation. The LTLf
layer remains the judge of what already happened; obligations govern what
must happen next.

## 5. The controller's four questions

- `admissible_actions(candidates, agent_state)`: cheap, tool-level.
  Per candidate: deepcopy grounding → hypothetical `called(t)` valuation →
  `probe` every monitor via snapshot/step/peek/restore. Rules whose
  formulas reference argument-level atoms (`arg_value`, `arg_has`, ...)
  are *excluded* here (args don't exist yet; optimistic, false-positive-
  free) and enforced only at validate time.
- `validate(action, agent_state)`: the hard pre-execution boundary, all
  rules, concrete args. Returns a structured `Decision`:
  - `ALLOW`;
  - `REQUIRE` (all violated rules recoverable in-session: missing facts /
    precedence): carries `requirements` and concrete `hints` ("call
    `verify_identity` to establish identity_verified"), the deterministic
    recovery channel. A predicate the
    evaluator derives (asserted by no tool) is expanded through the
    evaluator's optional `missing(predicate, ctx)` to the base facts still
    lacking and their tools; when nothing resolves, the hint says the name
    is a condition, not a tool;
  - `BLOCK` (hard: forbidden_when / at_most / custom): "do not retry".
- `next_decision(candidates, agent_state)`: the control handoff:
  - **Case C** (obligation due): `SYMBOLIC_EXECUTE` if args resolve via
    the spec's constant args / deterministic binder; else `NEURAL_CHOICE`
    restricted to the obligated tool.
  - **Case B** (exactly one admissible action): `SYMBOLIC_EXECUTE` iff the
    tool's `args_binder` (or an empty required-param schema) determines the
    arguments without semantic reasoning; else `NEURAL_CHOICE` over the
    singleton (model fills args only).
  - **Case A** (genuine freedom): `NEURAL_CHOICE` over the admissible set.
  - `DEAD_END` when nothing is admissible.
- `verify_return(action, result)`: the enforcement map on the return
  event; returns a `ReturnVerdict` naming the contracts whose assumption
  the result would falsify, `suppressed` on the enforcing side only.
- `update(action, result, return_verdict=)`: commit; ground the executed
  call event (with a `succeeded` arg), step all monitors, then, unless the
  return is suppressed, run the tool-result → fact extractors and
  invalidations, then discharge obligations. Tools never write controller
  state directly.

Filtering and validation share logic in v0 but are deliberately separate
APIs: filtering stays cheap; validate() will grow richer checks (argument
constraints, evidence, external predicates).

## 6. The runtime loop (`acuto/loop.py`)

Minimal by design; no procedural semantics live in the loop:

```
candidates = agent.available_actions(all tools)      # A_agent
decision   = controller.next_decision(candidates)     # ∩ A_contract
SYMBOLIC_EXECUTE → validate → execute → verify return → update both → notify agent
NEURAL_CHOICE    → model.generate(context, tools=admissible subset)
                   each proposed call → validate → execute/feedback
DEAD_END         → stop (v0; recovery policies later)
```

`FreeAgent` returns all tools as candidates (A_effective = A_contract).
A staged flow (`GraphFlow`) returns its state's candidate set instead;
the effective space is the intersection, on the same runtime.

## 7. Extension seams preserved (not implemented)

Design test: *can a local boolean predicate later become evidence-backed or
externally evaluated without rewriting the loop, backend, or controller?*

- **`PredicateEvaluator`**: all predicate reads go through it;
  `LocalPredicateEvaluator` (FactStore lookup) is the only v0 impl.
- **`Fact`**: stable identity, arbitrary `value`, optional `metadata`
  (source/timestamp today; provenance, expiry, dependencies later). Never
  assumed to be a raw boolean.
- **Explicit invalidation**: `assert_fact` / `invalidate_fact`; facts are
  not monotonic. Automatic dependency propagation deferred.
- **Structured decisions**: `Decision {ALLOW, BLOCK, REQUIRE}` +
  `StepDecision {NEURAL_CHOICE, SYMBOLIC_EXECUTE, DEAD_END}`; never a bool.
- **Tool-result → symbolic-event boundary**: `ToolResult` → extractors →
  facts; execution and symbolic interpretation stay separate.
- **`ContractBackend` protocol**: probe/step/finalize over valuations;
  the LTLf/DFA backend is one implementation.
- **Model adapter**: provider logic stays outside the runtime; neutral
  message format; Gemini REST + Mock today, Anthropic/OpenAI-compat next.

## 8. Deferred (explicitly out of v0)

Evidence graph; data-dependency tracking; external predicate verification;
semantic/LLM predicates; automatic evidence invalidation; SMT integration;
token-level constrained decoding; vLLM/SGLang; generalized policy IR;
NL→formal translation as a core dependency; GUI/sandbox/browser/MCP/
multi-agent/distributed; realizability lookahead via `contragent.formulas.sat`
(one-step probes can still walk into `G(¬goal)`-style dead ends; the sat
machinery exists upstream when we need it); latch policy (permanent goal
blocking on immutable-state violations).
