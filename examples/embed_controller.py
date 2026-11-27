"""Embed Acuto in a loop you own.

``examples/bank_demo.py`` lets ``acuto.Agent`` drive the loop. This
example is the other entry point: you keep your own message list and
your own model call, and Acuto supplies one ``acuto.Controller`` per task
that masks the tools, fast-forwards determined steps, verifies every call
before it runs, and verifies every result before the model sees it.

    python3 examples/embed_controller.py

No API key is needed: the model is scripted. Replace ``scripted_model``
with any callable that takes (messages, tool schemas) and returns tool
calls or text, and the loop below stays the same.
"""

from __future__ import annotations

import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
try:
    import contragent  # noqa: F401
except ModuleNotFoundError:  # development layout: sibling ContrAgent checkout
    sys.path.insert(0, str(_ROOT.parent / "ContrAgent"))

import acuto  # noqa: E402
from acuto.models import MockModel, ModelTurn, ToolCall  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Tools: what the agent CAN do.
# ---------------------------------------------------------------------------

world = {"fraud": True, "frozen": [], "refunds": []}
registry = acuto.ToolRegistry()


@registry.tool
def verify_identity(user_id: str) -> dict:
    "Verify the customer's identity."
    return {"verified": True, "user_id": user_id}


@registry.tool
def check_fraud(user_id: str) -> dict:
    "Run the fraud check for a customer."
    return {"fraud": world["fraud"], "user_id": user_id}


@registry.tool
def issue_refund(order_id: str, amount: float) -> dict:
    "Issue a refund for an order."
    world["refunds"].append((order_id, amount))
    return {"refunded": True}


@registry.tool
def freeze_account(user_id: str) -> dict:
    "Freeze the customer's account."
    world["frozen"].append(user_id)
    return {"frozen": True}


# ---------------------------------------------------------------------------
# 2. Contracts: what is ALLOWED and REQUIRED. Kept apart from the tools
#    and from the loop; verify() certifies the library before any run.
# ---------------------------------------------------------------------------

library = acuto.ContractLibrary(
    "refund-desk-v1",
    [
        acuto.action("issue_refund").requires("identity_verified", "fraud_checked").at_most(1),
        acuto.action("issue_refund").forbidden_when("fraud_detected"),
        acuto.after("verify_identity").asserts(
            "identity_verified", when=lambda r: r.output.get("verified")
        ),
        acuto.after("check_fraud")
        .asserts("fraud_checked")
        .asserts(
            "fraud_detected",
            when=lambda r: r.output.get("fraud"),
            metadata=lambda r: {"user_id": r.output.get("user_id")},
        ),
        acuto.when("fraud_detected").obligates(
            "freeze_account",
            binder=lambda ctx: {"user_id": ctx.facts.get("fraud_detected").metadata["user_id"]},
            desc="fraud detected: freeze the account immediately",
        ),
    ],
)
print(library.verify().render())

# ---------------------------------------------------------------------------
# 3. Your model. Scripted here; a real one is any function of the
#    messages and the (already masked) tool schemas.
# ---------------------------------------------------------------------------

scripted_model = MockModel(
    [
        ModelTurn(tool_calls=[ToolCall("issue_refund", {"order_id": "o1", "amount": 25.0})]),
        ModelTurn(tool_calls=[ToolCall("verify_identity", {"user_id": "u1"})]),
        ModelTurn(tool_calls=[ToolCall("check_fraud", {"user_id": "u1"})]),
        ModelTurn(tool_calls=[ToolCall("issue_refund", {"order_id": "o1", "amount": 25.0})]),
        ModelTurn(text="The account shows fraud, so no refund can be issued; it has been frozen."),
    ]
)

# ---------------------------------------------------------------------------
# 4. Your loop. Acuto owns mask → route → verify → commit; you own the rest.
# ---------------------------------------------------------------------------

controller = acuto.Controller(library, tools=registry)
messages: list[dict] = [{"role": "user", "content": "Refund order o1 for customer u1."}]

for step in range(12):
    decision = controller.route()  # stage 1 + 2: mask, then who acts
    if decision.kind is acuto.StepKind.DEAD_END:
        print(f"[{step}] dead end: {decision.reason}")
        break

    if decision.kind is acuto.StepKind.SYMBOLIC_EXECUTE:
        # Fast-forward: the procedure leaves one legal move, so Acuto runs
        # it. No model call. The call still crosses verify() inside execute().
        outcome = controller.execute(decision.action)
        print(f"[{step}] FAST-FORWARD {outcome.action} -> {outcome.feedback()}")
        messages.append({"role": "user", "content": "[auto] " + json.dumps(outcome.feedback())})
        continue

    schemas = registry.schemas(decision.actions)  # only the admissible tools
    turn = scripted_model.generate(messages, tools=schemas)
    print(f"[{step}] model saw {decision.actions}")

    if not turn.tool_calls:
        owed = controller.due_obligation()  # nothing may end while an obligation is due
        if owed is not None:
            outcome = controller.execute(owed)
            print(f"[{step}] FAST-FORWARD (owed) {outcome.action} -> {outcome.feedback()}")
            continue
        print(f"[{step}] final: {turn.text}")
        break

    for call in turn.tool_calls:
        outcome = controller.execute(call)  # stage 3: verify → run → commit
        tag = "ran" if outcome.executed else f"BLOCKED ({outcome.decision.kind.value})"
        print(f"[{step}] {tag} {call.name}{call.args} -> {outcome.feedback()}")
        messages.append(
            {"role": "tool", "name": call.name, "content": json.dumps(outcome.feedback())}
        )

print("\nend of session:", controller.finalize())
print("world:", world)
