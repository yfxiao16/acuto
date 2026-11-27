"""Acuto showcase: a bank card-replacement desk with a real SOP.

Run it (no API key needed; it drives a scripted model by default):

    python3 examples/bank_demo.py            # both scenarios
    python3 examples/bank_demo.py --fraud    # fraud scenario only
    python3 examples/bank_demo.py --happy    # happy path only
    python3 examples/bank_demo.py --live     # drive with Gemini (GEMINI_API_KEY)

The SOP (all enforced by Acuto, none of it prompted):

  1. Every case starts by loading the customer record.       (jump #1: Case B)
  2. replace_card REQUIRES identity_verified, fraud_checked,
     address_confirmed, and at most once per session.        (dynamic masking)
  3. change_address INVALIDATES address_confirmed.           (fact retraction)
  4. fraud_detected OBLIGATES freeze_account NOW.            (jump #2: Case C)
  5. replace_card is FORBIDDEN while fraud_detected.         (hard block)

What to watch in the output:

  * step 0 has NO model call: only lookup_customer is admissible and its
    argument is procedurally determined -> Acuto executes it symbolically.
  * the model never *sees* replace_card's schema until the three facts
    hold (dynamic per-step tool exposure); if it proposes it anyway, the
    hard pre-execution boundary blocks it and names the exact recovery
    tools (REQUIRE feedback).
  * on the fraud path, freeze_account happens between two model turns:
    obligation-driven symbolic execution, zero model involvement.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
try:
    import contragent  # noqa: F401
except ModuleNotFoundError:
    _kernel = _ROOT.parent / "ContrAgent"
    sys.path.insert(0, str(_kernel))

import acuto  # noqa: E402
from acuto.models import MockModel, ModelTurn, ToolCall  # noqa: E402

CUSTOMER = "C-1024"


# ---------------------------------------------------------------------------
# World + tools
# ---------------------------------------------------------------------------


def build_registry(world: dict) -> acuto.ToolRegistry:
    reg = acuto.ToolRegistry()

    @reg.tool(
        # Case B seam: the argument is procedurally determined from agent
        # state, so Acuto can execute this without a model decision.
        args_binder=lambda ctx: (
            {"customer_id": ctx.agent_state["customer_id"]} if ctx.agent_state else None
        )
    )
    def lookup_customer(customer_id: str) -> dict:
        "Load the customer's record. Every case starts here."
        return {"customer_id": customer_id, "name": "J. Doe", "found": True}

    @reg.tool
    def verify_identity(customer_id: str) -> dict:
        "Verify the customer's identity against the documents on file."
        return {"customer_id": customer_id, "verified": True}

    @reg.tool
    def check_fraud(customer_id: str) -> dict:
        "Run the fraud screen for this customer."
        return {"customer_id": customer_id, "fraud": world["fraud"]}

    @reg.tool
    def confirm_address(customer_id: str) -> dict:
        "Confirm the customer's mailing address."
        return {"customer_id": customer_id, "confirmed": True}

    @reg.tool
    def change_address(customer_id: str, new_address: str) -> dict:
        "Update the customer's mailing address."
        return {"customer_id": customer_id, "updated": True}

    @reg.tool
    def replace_card(customer_id: str, delivery: str = "mail") -> dict:
        "Order a replacement card."
        world["cards"].append(customer_id)
        return {"customer_id": customer_id, "ordered": True}

    @reg.tool
    def freeze_account(customer_id: str) -> dict:
        "Freeze the customer's account immediately."
        world["frozen"].append(customer_id)
        return {"customer_id": customer_id, "frozen": True}

    @reg.tool
    def escalate(reason: str) -> dict:
        "Escalate the case to a human specialist."
        return {"escalated": True}

    return reg


GATED = [
    "verify_identity",
    "check_fraud",
    "confirm_address",
    "change_address",
    "replace_card",
    "freeze_account",
    "escalate",
]


def build_contracts() -> list:
    contracts = [
        # 1. session preamble: nothing before the customer record is loaded
        *[acuto.action(t).requires("customer_loaded") for t in GATED],
        acuto.after("lookup_customer").asserts(
            "customer_loaded",
            when=lambda r: r.output.get("found"),
            metadata=lambda r: {"customer_id": r.output["customer_id"]},
        ),
        # 2. the replace_card gate
        acuto.action("replace_card")
        .requires("identity_verified", "fraud_checked", "address_confirmed")
        .at_most(1),
        acuto.after("verify_identity").asserts(
            "identity_verified", when=lambda r: r.output.get("verified")
        ),
        acuto.after("check_fraud")
        .asserts("fraud_checked")
        .asserts(
            "fraud_detected",
            when=lambda r: r.output.get("fraud"),
            metadata=lambda r: {"customer_id": r.output["customer_id"]},
        ),
        acuto.after("confirm_address").asserts(
            "address_confirmed", when=lambda r: r.output.get("confirmed")
        ),
        # 3. address change retracts the confirmation
        acuto.after("change_address").invalidates("address_confirmed"),
        # 4. fraud -> freeze NOW (obligation, executed by Acuto)
        acuto.when("fraud_detected").obligates(
            "freeze_account",
            binder=lambda ctx: (
                {"customer_id": ctx.facts.get("fraud_detected").metadata["customer_id"]}
                if ctx.facts.get("fraud_detected")
                else None
            ),
            desc="fraud detected: freeze the account immediately",
        ),
        # 5. no card replacement on a fraud-flagged account, ever
        acuto.action("replace_card").forbidden_when("fraud_detected"),
    ]
    return contracts


# ---------------------------------------------------------------------------
# Scripted models (deterministic; swap for GeminiModel with --live)
# ---------------------------------------------------------------------------


def happy_script() -> MockModel:
    c = {"customer_id": CUSTOMER}
    return MockModel(
        [
            # Models do this: propose the goal call even though its schema
            # was never exposed. The hard boundary catches it.
            ModelTurn(tool_calls=[ToolCall("replace_card", dict(c))]),
            ModelTurn(tool_calls=[ToolCall("verify_identity", dict(c))]),
            ModelTurn(tool_calls=[ToolCall("check_fraud", dict(c))]),
            ModelTurn(tool_calls=[ToolCall("confirm_address", dict(c))]),
            ModelTurn(tool_calls=[ToolCall("replace_card", dict(c))]),
            ModelTurn(text="Your replacement card is on its way."),
        ]
    )


def fraud_script() -> MockModel:
    c = {"customer_id": CUSTOMER}
    return MockModel(
        [
            ModelTurn(tool_calls=[ToolCall("verify_identity", dict(c))]),
            ModelTurn(tool_calls=[ToolCall("check_fraud", dict(c))]),
            # By now Acuto has already frozen the account. The model tries
            # the replacement anyway -> hard BLOCK (forbidden_when).
            ModelTurn(tool_calls=[ToolCall("replace_card", dict(c))]),
            ModelTurn(tool_calls=[ToolCall("escalate", {"reason": "fraud flag on account"})]),
            ModelTurn(
                text="I cannot replace the card: the fraud screen flagged this account, "
                "so it has been frozen and the case is escalated to a specialist."
            ),
        ]
    )


def run_scenario(name: str, *, fraud: bool, live: bool) -> None:
    print(f"\n{'=' * 70}\n  SCENARIO: {name}\n{'=' * 70}")
    world = {"fraud": fraud, "cards": [], "frozen": []}
    registry = build_registry(world)
    if live:
        from acuto.models import GeminiModel

        model = GeminiModel()
    else:
        model = fraud_script() if fraud else happy_script()
    library = acuto.ContractLibrary("card-replacement-v1", build_contracts())
    agent = acuto.Agent(
        model=model,
        tools=registry,
        contracts=library,
        flow=acuto.FreeFlow(state={"customer_id": CUSTOMER}),
        instructions="You are a bank service agent. Use the available tools to help the customer.",
    )
    agent.run(f"Customer {CUSTOMER} lost their card and wants a replacement.", verbose=True)
    print(f"\n  world after: cards={world['cards']} frozen={world['frozen']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--happy", action="store_true", help="happy path only")
    ap.add_argument("--fraud", action="store_true", help="fraud path only")
    ap.add_argument("--live", action="store_true", help="use Gemini instead of the script")
    args = ap.parse_args()
    both = not (args.happy or args.fraud)
    if args.happy or both:
        run_scenario("card replacement: happy path", fraud=False, live=args.live)
    if args.fraud or both:
        run_scenario("card replacement: fraud detected", fraud=True, live=args.live)


if __name__ == "__main__":
    main()
