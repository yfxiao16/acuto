"""The smallest Acuto agent: three tools, two contracts, one run.

    python3 examples/quickstart.py                # scripted model, no credentials
    OPENAI_API_KEY=... python3 examples/quickstart.py --model openai:gpt-5-mini

With ``--model`` (or a .env file next to you holding the provider's key)
the real model drives the conversation; otherwise a scripted model plays
the same turns, so the trace below shows what Acuto does either way:
the refund tool is hidden until the identity is verified, a premature
refund is refused with the recovery step named, and a second refund is
refused outright.
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
from acuto.models import MockModel, ModelTurn, ToolCall  # noqa: E402


def scripted_model() -> MockModel:
    return MockModel(
        [
            ModelTurn(tool_calls=[ToolCall("issue_refund", {"order_id": "123", "amount": 40.0})]),
            ModelTurn(tool_calls=[ToolCall("verify_identity", {"user_id": "u1"})]),
            ModelTurn(tool_calls=[ToolCall("issue_refund", {"order_id": "123", "amount": 40.0})]),
            ModelTurn(tool_calls=[ToolCall("issue_refund", {"order_id": "123", "amount": 40.0})]),
            ModelTurn(text="Refunded order 123 once; a second refund is not permitted."),
        ]
    )


def main(model_spec: str | None) -> None:
    model = acuto.models.resolve(model_spec) if model_spec else scripted_model()
    agent = acuto.Agent(model=model, instructions="You are a bank service agent.")

    @agent.tool
    def verify_identity(user_id: str) -> dict:
        "Verify the customer's identity."
        return {"verified": True}

    @agent.tool
    def issue_refund(order_id: str, amount: float) -> dict:
        "Issue a refund for an order."
        return {"refunded": True, "order_id": order_id}

    @agent.tool
    def send_receipt(order_id: str) -> dict:
        "Email the refund receipt."
        return {"sent": True}

    library = acuto.ContractLibrary(
        "refund-desk-v1",
        [
            acuto.action("issue_refund").requires("identity_verified").at_most(1),
            acuto.after("verify_identity").asserts(
                "identity_verified", when=lambda r: r.output["verified"]
            ),
        ],
    )
    library.verify()
    agent.attach(library)

    agent.run("Refund order #123 for customer u1.", verbose=True)


if __name__ == "__main__":
    spec = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--model":
        spec = sys.argv[2]
    main(spec)
