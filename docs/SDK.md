# Acuto SDK reference

Acuto is a framework for writing tool-using LLM agents whose procedure is
declared once, as contracts, and enforced at runtime. This page is the
reference for people who build on it. `README.md` has the quick start,
`DESIGN.md` the architecture, and `examples/` runnable programs.

## 1. The public surface

Everything is importable from `acuto`. The names fall into three tiers
(`acuto.__all__` lists them in this order), and most projects need only
the first.

| Tier | Names | Use it for |
|---|---|---|
| Write an agent | `Agent`, `models`, `Tool`, `ToolRegistry`, `ContractLibrary`, `action`, `after`, `when`, `ag`, `GraphFlow`, `FreeFlow`, `Frame`, `RunResult` | Declare tools, contracts, and an optional flow; let Acuto drive the loop. |
| Embed in your loop | `Controller`, `CallOutcome`, `as_action`, `ProposedAction`, `StepDecision`, `StepKind`, `Decision`, `DecisionKind`, `ReturnVerdict`, `ToolResult`, `run`, `iterate` | Keep your own loop and model call; Acuto masks, routes, verifies, commits. |
| Extend | `PredicateEvaluator`, `PredicateContext`, `LocalPredicateEvaluator`, `Fact`, `FactStore`, `Flow`, `FlowContext`, `CustomRule`, `Contract`, `CompiledContracts`, `compile_contracts`, `VerifyReport`, `ContractConflictError`, `SymbolicController`, `ContractBackend`, `LTLfBackend`, `ResidualPolicyCache`, obligation types, `ToolExecutor`, `Tracer` | Custom predicates, raw formulas, custom flows, the cache, the underlying controller. |

## 2. Five words

- A **contract** is one rule about the tool-call trace: "issue_refund
  requires identity_verified", "cancel at most once".
- A **fact** is something a tool result (or the task input) established:
  "identity_verified". Facts live in the controller's `FactStore`; tools
  never write them directly, declared extractors do.
- An **obligation** is a duty a fact creates: "once fraud_detected,
  freeze_account". Acuto discharges it itself when the arguments are
  determined.
- **Masking** exposes only the tools the contracts currently admit.
- **Fast-forward** executes the step when the contracts leave exactly one
  legal move with bound arguments, with no model call.

## 3. Tools

```python
registry = acuto.ToolRegistry()

@registry.tool                              # name, description, schema from the signature
def verify_identity(user_id: str) -> dict:
    "Verify the customer's identity."
    return {"verified": True, "user_id": user_id}

@registry.tool(name="freeze", description="Freeze the account.",
               args_binder=lambda ctx: {"user_id": ctx.facts.value("customer_id")})
def freeze_account(user_id: str) -> dict: ...
```

`@agent.tool` on an `Agent` is the same decorator. A tool's `args_binder`
is a deterministic function `PredicateContext -> dict | None`: when the
contracts narrow the action space to this one tool and the binder returns
a dict, Acuto executes the call without asking the model. Tool exceptions
become `ToolResult(ok=False, error=...)` observations, never crashes.
`ToolExecutor(pre_hooks=..., post_hooks=...)` wraps every execution,
including fast-forwarded ones.

## 4. Contracts

### 4.1 The fluent DSL

```python
acuto.action("issue_refund").requires("identity_verified", "fraud_checked")
acuto.action("issue_refund").forbidden_when("account_frozen")
acuto.action("issue_refund").requires_before("check_policy").at_most(1)

acuto.after("verify_identity").asserts("identity_verified",
                                       when=lambda r: r.output["verified"])
acuto.after("lookup").asserts(lambda r: f"order:{r.output['id']}",    # one fact per entity
                              value=lambda r: r.output, metadata=lambda r: {...})
acuto.after("change_identity_document").invalidates("identity_verified")

acuto.when("fraud_detected").obligates(
    "freeze_account",
    binder=lambda ctx: {"user_id": ctx.facts.get("fraud_detected").metadata["user_id"]},
    deadline="next",          # or "eventually"
    desc="fraud detected: freeze the account immediately",
)
```

`action(...)` rules are guarantees over the trace and compile to LTLf
monitored by a DFA. `after(...)` rules are the tool-result → fact
boundary; they run only on successful results, unless an assertion is
made with `on_error=True`, which a call that ran and returned an error
makes too. `when(...)` rules are obligations, discharged by a successful
call of the tool, or by any call that ran with `discharged_on_error=True`.
The two flags mirror `called(tool)` in a contract, which holds of every
call that ran, error or not. A `requires` violation is recoverable (`REQUIRE`: the model
is told which tool establishes the missing fact); `forbidden_when` and
`at_most` are hard (`BLOCK`).

### 4.2 Raw formulas and assume-guarantee contracts

```python
from contragent.formulas.formula import Atom, G, Not, Implies

acuto.CustomRule(formula=G(Implies(Atom("called", "a"), Not(Atom("called", "b")))),
                 name="a-then-never-b", masking=False)      # masking=False: soft tier
acuto.ag(assumption=G(Not(Atom("fact", "channel_closed"))),   # what the environment keeps
         guarantee=G(Not(Atom("called", "audit_log"))),       # what the agent keeps meanwhile
         name="no-audit-while-open")
```

An assumption is written over environment predicates only (`fact(...)`,
results). A tool result that would falsify one is *suppressed*: no facts
are written and the model receives a withheld-result error instead of the
result. `compile_contracts` rejects assumptions over the agent's own calls
or with unbounded eventualities.

### 4.3 Where a contract library lives

1. **Inline**, next to the tools, for small agents (the README example).
2. **A module of its own** (`contracts.py` exporting one
   `acuto.ContractLibrary("name-v1", [...])`), verified at import time:

   ```python
   library = acuto.ContractLibrary("refund-desk-v1", RULES)
   library.verify()      # raises ContractConflictError on a definitive conflict
   ```

3. **A ContrAgent YAML file**, shared with ContrAgent's own tools:

   ```python
   library = acuto.ContractLibrary.from_contragent("policies/retail.yaml", agent="agent")
   ```

   Each YAML contract's assumptions and guarantees are preserved as one
   `ag` contract; non-deterministic (`sto`) constraints are skipped.
   `library.to_contragent()` goes the other way, and
   `library.check_conflicts()` runs ContrAgent's minimal-unsat-core
   analysis on the library.

`verify()` certifies each contract satisfiable and falsifiable, each
assumption satisfiable, and the library jointly satisfiable; `None`
answers mean the search budget ran out, not an error. A library compiles
once and is reused across runs; control state is per session.

## 5. Flows

A flow is the application's own organization of the task: which tools
are candidates now and how the model's context is kept. Contracts stay
in the library, never in the flow.

```python
flow = acuto.GraphFlow(start="triage", state={"customer_id": "C-1"})
flow.state("triage", tools=["lookup_customer"],
           next=lambda ctx: "work" if ctx.facts.truthy("customer_loaded") else None)
flow.state("work", tools=["verify_identity", "issue_refund", "submit"],
           next=lambda ctx: "done" if ctx.facts.truthy("submitted") else None)
flow.state("done", tools=[], terminal=True)
```

The effective toolset each step is `A_eff = A_flow ∩ A_contract`. A
flow with a terminal state does not end on a text-only answer; the model
is nudged back to finish the procedure. `FreeFlow` (the default) offers
every tool every step. Implement the `Flow` protocol for anything else.

## 6. Models

`acuto.models.resolve("provider:model")` covers `anthropic`, `openai`
(and any OpenAI-compatible endpoint via `base_url=`), `gemini`, and
`bedrock`. Credentials follow the provider SDKs' convention: the key is
read from the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY`, `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`, or the
variable named by `api_key_env=`), then from a `.env` file in the
working directory, and `api_key=` overrides both. Custom OpenAI-compatible
endpoints use `api_key=` or `api_key_env=` when specified. When neither is
specified, requests to those endpoints omit authorization. Any object with
`generate(messages, tools, system) -> ModelTurn` is a model;
`MockModel` replays a script for tests. `ReActModel` wraps a model that
has no native tool calling in a text protocol.

## 7. Running

```python
agent = acuto.Agent(model, tools=registry, contracts=library, flow=flow,
                    instructions="...", max_steps=24,
                    control_mode="full",          # "full" | "mask" | "passive"
                    mask_granularity="step",      # "step" | "phase" | "hint"
                    cache=True)                   # residual policy cache across runs
result = agent.run("Refund order #123", facts={"customer_id": "C-1"},
                   verbose=True)                  # print every decision as it happens
for frame in agent.stream("..."):                 # one Frame per step
    print(frame.kind, frame.exposed, frame.masked, frame.blocked)
acuto.console.render(result)                      # replay a finished run
```

`verbose=True` (also on `acuto.run`) prints a banner, one block per
step naming who acted (`FAST-FORWARD`, `MODEL`, `DEAD END`) and what
followed (`masked`, `proposes`, `ran`, `REQUIRE`, `BLOCK`, `WITHHELD`,
facts, obligations), and a summary. Colour is on for terminals and off
under `NO_COLOR`; `Console(stream, color=False)` writes plain text
anywhere.

`RunResult` carries `status`, `final_text`, counts (`model_calls`,
`symbolic_steps`, `blocked_proposals`), the `finalize()` report, the
tracer, and a wall-clock breakdown. Pass `trace_path=` to write the
trace as JSONL, and `auditor=library.auditor()` to score compliance
condition-independently.

## 8. Embedding: `acuto.Controller`

One `Controller` is one session. Build it from a library (or a list of
rules), optionally with the tools, and drive it from any loop:

| Method | Stage | Returns |
|---|---|---|
| `mask(candidates=None, state=None)` | mask | admissible tool names |
| `route(candidates=None, state=None)` | mask + route | `StepDecision`: `SYMBOLIC_EXECUTE` with the bound call, `NEURAL_CHOICE` with the tools to expose, or `DEAD_END` |
| `verify(call, state=None)` | verify | `Decision`: `ALLOW`, `REQUIRE` (with hints), `BLOCK` |
| `commit(call, result, state=None)` | verify the return | the `ToolResult` the model may see (a withheld-result error if suppressed) |
| `execute(call, state=None)` | all three | `CallOutcome` with `decision`, `result`, `shown`, `feedback()` |
| `due_obligation(state=None)` | floor-yield gate | the owed call, or `None` |
| `assert_fact(name, value)` | external facts | `False` if suppressed |
| `finalize()` | session end | violations, pending obligations, broken assumptions |

`call` may be a `ProposedAction`, a `(tool, args)` pair, a dict, or a
model `ToolCall`. Loops that execute tools elsewhere call `verify` before
and `commit` after; `execute` needs `tools=`. The built-in loop
(`acuto.loop.iterate`) runs on this same object, so `acuto.run(...,
controller=my_controller)` shares the session.

## 9. Extension points

- **Predicates.** A `PredicateEvaluator` resolves `fact(...)` atoms;
  subclass `LocalPredicateEvaluator` to derive predicates from other
  facts or application state, and override `missing()` so `REQUIRE`
  feedback names the base tools to call.
- **Action-relative context.** `context_deriver(action, facts) -> dict`
  computes `target_*` context for the call being judged (which
  reservation a cancel names); contracts over it are enforced at verify.
- **Argument binding.** `args_binder` on a tool, or `binder=` on an
  obligation, turns facts into arguments deterministically.
- **Flows.** Implement the `Flow` protocol for your own context and
  candidate policy.
- **Models.** Implement `generate(messages, tools, system)`.
- **Backends.** `ContractBackend` (probe/step/finalize over valuations);
  `LTLfBackend` is the shipped implementation.
- **Tracing.** Every decision is a `Tracer` record (`action/masked`,
  `action/blocked`, `action/symbolic`, `return/withheld`, ...); use
  `tracer.by_kind(...)` or `trace_path=`.

## 10. Contract semantics in one paragraph

Every executed call, whether the model proposed it or the controller
scheduled it, crosses `verify` first. The call event always commits. The
return event is checked as a whole against every assumption before its
facts land; a breaking return is withheld from the model and writes
nothing. At session end, pending obligations and contracts left
unsatisfied are violations. Masking is a per-step convenience over the
same verdicts; the guarantee comes from `verify`.
