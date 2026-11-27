"""Bring your own model: any object with ``generate`` is a model.

    python3 examples/custom_model.py

Acuto ships adapters for Anthropic, OpenAI-compatible endpoints, Gemini
and Bedrock (``acuto.models.resolve``). To use a model behind your own
API, write a class with one method, ``generate(messages, tools, system)``,
that returns a ``ModelTurn``: the text the model said, the tool calls it
made, or both. Acuto does the rest: it decides which tools the model
sees, checks every call it makes, and runs the calls the contracts
determine without asking it.

``MyModel`` below stands in for a call to your service; it answers from
the conversation so the example runs without credentials.
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
try:
    import contragent  # noqa: F401
except ModuleNotFoundError:  # development layout: sibling ContrAgent checkout
    sys.path.insert(0, str(_ROOT.parent / "ContrAgent"))

import acuto  # noqa: E402
from acuto.models import ModelTurn, ToolCall  # noqa: E402


class MyModel:
    """Replace the body of ``generate`` with a request to your API.

    ``messages`` is the conversation in Acuto's neutral format (roles
    ``user``, ``assistant``, ``tool``), ``tools`` the JSON schemas of the
    tools the contracts admit at this step, ``system`` the instructions.
    """

    def generate(
        self, messages: list[dict], tools: list[dict], system: str | None = None
    ) -> ModelTurn:
        offered = {t["name"] for t in tools}
        called = {m["name"] for m in messages if m["role"] == "tool"}
        for name, args in (
            ("verify_identity", {"user_id": "u1"}),
            ("issue_refund", {"order_id": "123", "amount": 40.0}),
        ):
            if name in offered and name not in called:
                return ModelTurn(tool_calls=[ToolCall(name, args, id=f"call_{len(messages)}")])
        return ModelTurn(text="The refund for order 123 has been issued.")


def main() -> None:
    agent = acuto.Agent(model=MyModel(), instructions="You are a bank service agent.")

    @agent.tool
    def verify_identity(user_id: str) -> dict:
        "Verify the customer's identity."
        return {"verified": True}

    @agent.tool
    def issue_refund(order_id: str, amount: float) -> dict:
        "Issue a refund for an order."
        return {"refunded": True, "order_id": order_id}

    agent.attach(
        acuto.ContractLibrary(
            "refund-desk-v1",
            [
                acuto.action("issue_refund").requires("identity_verified").at_most(1),
                acuto.after("verify_identity").asserts(
                    "identity_verified", when=lambda r: r.output["verified"]
                ),
            ],
        )
    )
    result = agent.run("Refund order #123 for customer u1", verbose=True)
    print(result.final_text)


if __name__ == "__main__":
    main()
