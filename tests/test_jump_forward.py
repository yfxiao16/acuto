"""Symbolic control handoff: obligations preempt neural control; a
single admissible action with determined arguments executes without the
model."""

from __future__ import annotations

import acuto
from tests.conftest import execute_and_commit


def test_obligation_triggers_symbolic_execution(controller, registry, world):
    world["fraud"] = True
    execute_and_commit(controller, registry, "check_fraud", {"user_id": "u9"})

    decision = controller.next_decision(registry.names())
    assert decision.kind is acuto.StepKind.SYMBOLIC_EXECUTE
    assert decision.action.tool == "freeze_account"
    assert decision.action.args == {"user_id": "u9"}  # bound from fact metadata

    # Execute + commit discharges the obligation; control returns to neural.
    execute_and_commit(controller, registry, "freeze_account", decision.action.args)
    assert not controller.obligations.pending()
    after = controller.next_decision(registry.names())
    assert after.kind is acuto.StepKind.NEURAL_CHOICE


def test_obligation_without_binder_narrows_to_singleton(registry):
    contracts = [
        acuto.when("fraud_detected").obligates("freeze_account", desc="freeze on fraud"),
    ]
    controller = acuto.SymbolicController(contracts, registry=registry)
    controller.assert_fact("fraud_detected", True)

    decision = controller.next_decision(registry.names())
    # freeze_account requires user_id (no binder) -> the LLM decides,
    # but only over the obligated action.
    assert decision.kind is acuto.StepKind.NEURAL_CHOICE
    assert decision.actions == ["freeze_account"]


def test_single_admissible_action_with_binder_jumps(registry):
    reg = acuto.ToolRegistry()

    @reg.tool(args_binder=lambda ctx: {"user_id": ctx.facts.value("customer_id")})
    def login_user(user_id: str) -> dict:
        "Log the user in."
        return {"logged_in": True}

    @reg.tool
    def issue_refund(order_id: str) -> dict:
        "Issue a refund."
        return {"refunded": True}

    contracts = [acuto.action("issue_refund").requires("logged_in")]
    controller = acuto.SymbolicController(contracts, registry=reg)
    controller.assert_fact("customer_id", "u42")

    # issue_refund is masked; only login_user is admissible and its args
    # are procedurally determined -> jump-forward.
    decision = controller.next_decision(reg.names())
    assert decision.kind is acuto.StepKind.SYMBOLIC_EXECUTE
    assert decision.action.tool == "login_user"
    assert decision.action.args == {"user_id": "u42"}


def test_pending_obligation_reported_at_finalize(registry):
    contracts = [acuto.when("fraud_detected").obligates("freeze_account", desc="freeze on fraud")]
    controller = acuto.SymbolicController(contracts, registry=registry)
    controller.assert_fact("fraud_detected", True)

    report = controller.finalize()
    assert report["pending_obligations"] == ["freeze on fraud"]


def test_eventually_obligation_floor_yield_gate(registry):
    """deadline='eventually' does not preempt, but a text-only answer may
    not end the run while it is pending — Acuto discharges it (jump) at
    the floor-yield boundary."""
    from acuto.models import MockModel, ModelTurn, ToolCall

    contracts = [
        acuto.after("check_fraud").asserts(
            "fraud_case_opened", metadata=lambda r: {"user_id": r.output.get("user_id")}
        ),
        acuto.when("fraud_case_opened").obligates(
            "freeze_account",
            deadline="eventually",
            binder=lambda ctx: {"user_id": ctx.facts.get("fraud_case_opened").metadata["user_id"]},
            desc="verify-after-modify style follow-up",
        ),
    ]
    model = MockModel(
        [
            ModelTurn(tool_calls=[ToolCall("check_fraud", {"user_id": "u5"})]),
            # Model tries to finish immediately — the gate must fire first.
            ModelTurn(text="All done!"),
            ModelTurn(text="All done!"),
        ]
    )
    result = acuto.run("case for u5", model=model, tools=registry, contracts=contracts)
    assert result.status == "completed"
    assert result.symbolic_steps == 1  # freeze executed at the yield boundary
    assert not result.finalize["pending_obligations"]
    # And it did NOT preempt: the tool ran only after the model tried to finish.
    kinds = [r["kind"] for r in result.tracer.records if r["kind"] == "action/symbolic"]
    assert len(kinds) == 1


def test_a_text_turn_cannot_end_the_session_while_a_next_obligation_is_pending():
    reg = acuto.ToolRegistry()

    @reg.tool
    def open_case() -> dict:
        "Open a case."
        return {"opened": True}

    @reg.tool
    def close_case(case_id: str) -> dict:
        "Close a case."
        return {"closed": case_id}

    contracts = [
        acuto.after("open_case").asserts("case_open"),
        acuto.when("case_open").obligates("close_case", desc="close the case next"),
    ]

    def gen(messages, tools, system):
        if not any(m.get("role") == "tool" for m in messages):
            return acuto.models.ModelTurn(tool_calls=[acuto.models.ToolCall("open_case", {})])
        if any("close the case next" in str(m.get("content")) for m in messages[-1:]):
            return acuto.models.ModelTurn(
                tool_calls=[acuto.models.ToolCall("close_case", {"case_id": "c1"})]
            )
        return acuto.models.ModelTurn(text="done")

    result = acuto.run(
        "Open a case.", model=acuto.models.MockModel([gen] * 6), tools=reg, contracts=contracts
    )
    # The case_id cannot be bound, so the model is handed close_case alone;
    # its premature "done" is held back with the obligation named, and it
    # then closes the case.
    assert result.tracer.by_kind("controller/nudge")
    assert result.status == "completed"
    assert result.finalize["pending_obligations"] == []


def test_a_call_that_ran_and_errored_can_trigger_and_discharge_an_obligation():
    from acuto.decisions import ProposedAction
    from acuto.tools import ToolResult

    def ran(tool, ok):
        return ProposedAction(tool, {}), ToolResult(
            tool=tool, args={}, ok=ok, error=None if ok else "boom"
        )

    contracts = [
        acuto.after("modify").asserts("modified", on_error=True),
        acuto.when("modified").obligates(
            "verify", desc="verify after modify", discharged_on_error=True
        ),
        acuto.after("open_case").asserts("case_open"),
        acuto.when("case_open").obligates("close_case", desc="close the case"),
    ]
    controller = acuto.SymbolicController(contracts)
    # An errored modify ran, so it triggers; an errored verify ran, so it discharges.
    controller.update(*ran("modify", False))
    assert [o.tool for o in controller.obligations.pending()] == ["verify"]
    controller.update(*ran("verify", False))
    assert not controller.obligations.pending()
    # Without the flags, only a successful call counts.
    controller.update(*ran("open_case", False))
    assert not controller.obligations.pending()
    controller.update(*ran("open_case", True))
    controller.update(*ran("close_case", False))
    assert [o.tool for o in controller.obligations.pending()] == ["close_case"]
