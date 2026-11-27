# Acuto

**A**gentic **C**ontract-based r**u**n**t**ime **O**rchestration: contract-based symbolic control of LLM agents.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/yfxiao16/acuto/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/yfxiao16/acuto/actions/workflows/ci.yml/badge.svg)](https://github.com/yfxiao16/acuto/actions/workflows/ci.yml)

Declare a procedure once, as contracts over the agent's tool calls.
Acuto hides the tools the contracts rule out, executes the steps they
determine without asking the model, and checks every call before it
runs.

Acuto (Agentic Contract-based Runtime Orchestration) is a symbolic
control layer for LLM agents. A library of assume-guarantee contracts in
ALTL<sub>f</sub> over the tool-call trace compiles to deterministic
automata whose joint state the framework walks as a residual policy
graph: each node carries the tools the contracts still admit and, when
only one remains, the call itself. Three mechanisms read that graph
before every step. **Masking** exposes only the admissible tools.
**Fast-forward** executes a step the contracts determine, or an
obligation that has fallen due, with no model call. **Verification**
checks every call before it executes, whether the model proposed it or
the framework scheduled it, and every result before the model sees it.
Acuto drives the agent loop itself, so the guarantee holds by
construction: every executed action crosses the same boundary.

## Installation

```bash
pip install acuto                # Python 3.10+
```

`pip` installs the `contragent` runtime dependency automatically. To
work on Acuto itself, clone this repository and run:

```bash
pip install -e ".[dev]"
```

## Quick start

An agent is a model, its tools, and a contract library. Tools say what
the agent *can* do; contracts say what is *allowed and required*; Acuto
drives the loop. Register tools as synchronous functions.

```python
import acuto

agent = acuto.Agent(model=acuto.models.resolve("openai:gpt-5-mini"),
                    instructions="You are a bank service agent.")

@agent.tool
def verify_identity(user_id: str) -> dict:
    "Verify the customer's identity."
    return {"verified": True}

@agent.tool
def issue_refund(order_id: str, amount: float) -> dict:
    "Issue a refund for an order."
    return {"refunded": True}

agent.attach(acuto.ContractLibrary("refund-desk-v1", [
    acuto.action("issue_refund").requires("identity_verified").at_most(1),
    acuto.after("verify_identity").asserts("identity_verified",
                                           when=lambda r: r.output["verified"]),
]))

agent.run("Refund order #123 for customer u1", verbose=True)
```

Until the identity is verified the model does not see `issue_refund`; a
premature refund is refused with the recovery step named; a second
refund is refused outright. A contract is a rule about the trace, a
fact is what a tool result established, and an obligation
(`acuto.when(fact).obligates(tool)`) is a duty a fact creates, which
Acuto discharges itself when its arguments are determined. `verbose=True`
prints every decision as it happens.

The model key is read from the environment (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or the AWS pair), then from a
`.env` in the working directory. [`examples/`](https://github.com/yfxiao16/acuto/tree/main/examples/) runs the same
programs with a scripted model and no credentials:
[`quickstart.py`](https://github.com/yfxiao16/acuto/blob/main/examples/quickstart.py) is the program above,
[`bank_demo.py`](https://github.com/yfxiao16/acuto/blob/main/examples/bank_demo.py) shows an obligation being
fast-forwarded and a forbidden call refused, and
[`embed_controller.py`](https://github.com/yfxiao16/acuto/blob/main/examples/embed_controller.py) drives Acuto from
a loop of its own.

## Use your own model

Acuto talks to the provider with your key; there is no Acuto service in
between and no provider SDK to install.

```python
import acuto

acuto.models.resolve("anthropic:claude-sonnet-5-5")          # ANTHROPIC_API_KEY
acuto.models.resolve("openai:gpt-5-mini")                    # OPENAI_API_KEY
acuto.models.resolve("gemini:gemini-2.5-flash")              # GEMINI_API_KEY
acuto.models.resolve("bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0")  # AWS keys

# any OpenAI-compatible endpoint: a hosted one, or vLLM / SGLang on your machine
acuto.models.resolve("openai:deepseek-chat",
                     base_url="https://api.deepseek.com/v1",
                     api_key_env="DEEPSEEK_API_KEY")

# local OpenAI-compatible server without authentication
acuto.models.resolve("openai:local-model", base_url="http://localhost:8000/v1")
```

For the default endpoints, credentials come from the environment variables
shown, then from a `.env` file in the working directory. For a custom
OpenAI-compatible endpoint, pass `api_key=` or `api_key_env=` if it requires
authentication. Acuto sends the request without an authorization header
when both are omitted. An explicit `api_key=` overrides the environment. For a model
behind your own API, write a class with one method,
`generate(messages, tools, system)`, that returns a `ModelTurn`:
[`examples/custom_model.py`](https://github.com/yfxiao16/acuto/blob/main/examples/custom_model.py).

## Long runs

A long task outgrows the context window and outlives the process. The
control state (monitors, facts, obligations) is kept apart from the
messages, so both can be handled without changing what is allowed or
required.

```python
from acuto import Agent, Checkpoint, MaskCompactor

agent = Agent(model=model, compactor=MaskCompactor(threshold=100_000))
agent.run(task, checkpoint_path="run.json")        # state written before every step

# after a crash or a kill: continue from the step that was cut off
agent.run(task, resume=Checkpoint.load("run.json"))
```

A compactor moves old tool results out of the model's view once the
context passes its threshold; the model reads them back with the
`acuto_recall` tool. A resumed run replays its journal into a fresh
controller, so no call that completed is made again. A call that was
sent when the run died, to a tool not marked `idempotent`, is not
repeated unless you pass `on_in_flight="redo"`.
[`examples/long_run.py`](https://github.com/yfxiao16/acuto/blob/main/examples/long_run.py) shows a run that dies and
is resumed; [docs/LONG_RUNS.md](https://github.com/yfxiao16/acuto/blob/main/docs/LONG_RUNS.md) has the design and
its guarantees.

## Using Acuto from your own loop

`acuto.Agent` is a reference loop over `acuto.Controller`, one session of
control any loop can drive. `route()` masks the tools and says who acts
next; `verify(call)` is the boundary every call crosses; `commit(call,
result)` verifies the result and advances the state; `execute(call)`
chains the three when Acuto runs the tools.

```python
controller = acuto.Controller(library, tools=registry)
while True:
    decision = controller.route()
    if decision.kind is acuto.StepKind.SYMBOLIC_EXECUTE:
        controller.execute(decision.action)            # no model call
        continue
    turn = my_model(messages, tools=registry.schemas(decision.actions))
    for call in turn.tool_calls:
        outcome = controller.execute(call)             # verify → run → commit
        messages.append(tool_message(outcome.feedback()))
```

## Documentation

- [SDK reference](https://github.com/yfxiao16/acuto/blob/main/docs/SDK.md): the public surface in three tiers, the
  contract DSL, where a contract library lives, flows, model adapters,
  and the extension points.
- [Design](https://github.com/yfxiao16/acuto/blob/main/docs/DESIGN.md) covers the runtime architecture.

## Citation

Cite the software using [CITATION.cff](https://github.com/yfxiao16/acuto/blob/main/CITATION.cff).

## License

Acuto is licensed under [Apache-2.0](https://github.com/yfxiao16/acuto/blob/main/LICENSE).
See [CONTRIBUTING.md](https://github.com/yfxiao16/acuto/blob/main/CONTRIBUTING.md)
for the development workflow.
