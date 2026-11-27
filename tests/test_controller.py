"""The Controller facade: one session of control driven step by step from
a loop Acuto does not own, and the built-in loop running on the same
object."""

from __future__ import annotations

import pytest
from contragent.formulas.formula import Atom, G, Not

import acuto
from acuto.models import MockModel, ModelTurn, ToolCall


def _custom_loop(controller: acuto.Controller, model: MockModel, registry: acuto.ToolRegistry):
    """A minimal loop a third party might write around Acuto: it owns the
    messages and the model call; Acuto owns mask, route, verify, commit."""
    messages: list[dict] = [{"role": "user", "content": "refund order o1 for u1"}]
    executed: list[str] = []
    for _ in range(12):
        decision = controller.route()
        if decision.kind is acuto.StepKind.DEAD_END:
            return "dead_end", executed, messages
        if decision.kind is acuto.StepKind.SYMBOLIC_EXECUTE:
            outcome = controller.execute(decision.action)
            assert outcome.executed
            executed.append(f"auto:{outcome.action.tool}")
            messages.append({"role": "user", "content": f"[auto] {outcome.feedback()}"})
            continue
        turn = model.generate(messages, tools=registry.schemas(decision.actions))
        if not turn.tool_calls:
            if controller.due_obligation() is not None:
                outcome = controller.execute(controller.due_obligation())
                executed.append(f"auto:{outcome.action.tool}")
                continue
            return "completed", executed, messages
        for call in turn.tool_calls:
            outcome = controller.execute(call)
            executed.append(("ran:" if outcome.executed else "blocked:") + call.name)
            messages.append({"role": "tool", "name": call.name, "content": str(outcome.feedback())})
    return "max_steps", executed, messages


def test_custom_loop_masks_blocks_and_recovers(registry, contracts, world):
    controller = acuto.Controller(contracts, tools=registry)
    model = MockModel(
        [
            ModelTurn(tool_calls=[ToolCall("issue_refund", {"order_id": "o1", "amount": 10.0})]),
            ModelTurn(tool_calls=[ToolCall("verify_identity", {"user_id": "u1"})]),
            ModelTurn(tool_calls=[ToolCall("check_fraud", {"user_id": "u1"})]),
            ModelTurn(tool_calls=[ToolCall("issue_refund", {"order_id": "o1", "amount": 10.0})]),
            ModelTurn(text="done"),
        ]
    )
    status, executed, messages = _custom_loop(controller, model, registry)
    assert status == "completed"
    # The first turn never saw issue_refund (masked), and the premature
    # proposal was refused with REQUIRE feedback naming the recovery tool.
    assert "issue_refund" not in model.calls[0]["tools"]
    assert executed == [
        "blocked:issue_refund",
        "ran:verify_identity",
        "ran:check_fraud",
        "ran:issue_refund",
    ]
    assert "verify_identity" in messages[1]["content"]
    assert world["refunds"] == [("o1", 10.0)]
    report = controller.finalize()
    assert report["ltlf_violations"] == [] and report["pending_obligations"] == []


def test_custom_loop_fast_forwards_obligation(registry, contracts, world):
    world["fraud"] = True
    controller = acuto.Controller(contracts, tools=registry)
    model = MockModel(
        [
            ModelTurn(tool_calls=[ToolCall("verify_identity", {"user_id": "u1"})]),
            ModelTurn(tool_calls=[ToolCall("check_fraud", {"user_id": "u1"})]),
            ModelTurn(text="I'm done."),
        ]
    )
    status, executed, _ = _custom_loop(controller, model, registry)
    assert status == "completed"
    # The obligation was discharged by the controller, with no model call.
    assert "auto:freeze_account" in executed
    assert world["frozen"] == ["u1"]
    assert model.call_count == 3


def test_verify_and_commit_around_external_execution(registry, contracts, world):
    """A harness that runs tools itself calls verify() before and commit()
    after; mask/route/verify/commit work without a registry at all."""
    controller = acuto.Controller(contracts)
    assert "issue_refund" not in controller.mask(registry.names())
    call = ("issue_refund", {"order_id": "o1", "amount": 1.0})
    assert controller.verify(call).kind is acuto.DecisionKind.REQUIRE
    with pytest.raises(RuntimeError):
        controller.execute(call)

    ran = acuto.ToolExecutor(registry).execute(
        acuto.as_action(("verify_identity", {"user_id": "u1"}))
    )
    shown = controller.commit(("verify_identity", {"user_id": "u1"}), ran)
    assert shown is ran
    assert controller.facts.truthy("identity_verified")


def test_commit_withholds_a_return_that_breaks_an_assumption():
    """The return event is verified: a result whose facts would falsify a
    contract's assumption is suppressed, writes nothing, and the model is
    handed a withheld-result error instead."""
    reg = acuto.ToolRegistry()

    @reg.tool
    def probe_channel() -> dict:
        "Probe the channel."
        return {"open": False}

    @reg.tool
    def audit_log() -> dict:
        "Write to the audit log."
        return {"ok": True}

    contract = acuto.ag(
        G(Not(Atom("fact", "channel_closed"))),  # the environment keeps the channel open
        G(Not(Atom("called", "audit_log"))),
        name="no-audit-while-open",
    )
    extractor = acuto.after("probe_channel").asserts(
        "channel_closed", when=lambda r: not r.output["open"]
    )
    controller = acuto.Controller([contract, extractor], tools=reg)
    outcome = controller.execute(("probe_channel", {}))
    assert outcome.executed and outcome.withheld
    assert outcome.result.ok and not outcome.shown.ok
    assert "withheld" in outcome.feedback()["error"]
    assert "channel_closed" not in controller.facts
    assert controller.tracer.by_kind("return/withheld")


def test_as_action_accepts_common_call_shapes():
    tc = ToolCall("t", {"a": 1})
    for call in (
        acuto.ProposedAction("t", {"a": 1}),
        ("t", {"a": 1}),
        {"name": "t", "args": {"a": 1}},
        tc,
    ):
        action = acuto.as_action(call)
        assert (action.tool, action.args) == ("t", {"a": 1})

    class FrameworkCall:
        name = "t"
        arguments = {"a": 1}

    assert acuto.as_action(FrameworkCall()).args == {"a": 1}
    with pytest.raises(TypeError):
        acuto.as_action(42)


def test_builtin_loop_runs_on_the_facade(registry, contracts, world):
    """acuto.run accepts a Controller and drives it through the same
    methods a custom loop uses; the run's facts are the facade's facts."""
    world["fraud"] = True
    controller = acuto.Controller(contracts, tools=registry, facts={"customer_id": "u1"})
    model = MockModel(
        [
            ModelTurn(tool_calls=[ToolCall("verify_identity", {"user_id": "u1"})]),
            ModelTurn(tool_calls=[ToolCall("check_fraud", {"user_id": "u1"})]),
            ModelTurn(text="done"),
        ]
    )
    result = acuto.run("refund", model=model, tools=registry, controller=controller)
    assert result.status == "completed"
    assert result.symbolic_steps == 1 and world["frozen"] == ["u1"]
    assert result.controller is controller.symbolic
    assert controller.facts.value("customer_id") == "u1"


def test_initial_facts_and_external_facts_go_through_the_same_funnel(registry, contracts):
    controller = acuto.Controller(contracts, tools=registry, facts={"fraud_checked": True})
    assert controller.facts.truthy("fraud_checked")
    assert controller.assert_fact("identity_verified") is True
    # Both prerequisites hold: issue_refund is admissible now.
    assert "issue_refund" in controller.mask()
