"""A fast-forward prediction that verification blocks is withheld and the
model decides the step: the blocked call never reaches the environment and
the session continues (it used to end as a dead end)."""

from __future__ import annotations

import acuto
from acuto import control
from acuto.decisions import Decision, DecisionKind, ProposedAction, StepDecision
from acuto.models.base import ModelTurn, ToolCall
from acuto.models.mock import MockModel


def test_blocked_prediction_falls_back_to_the_model(monkeypatch):
    calls = []
    model = MockModel(
        [
            ModelTurn(
                tool_calls=[ToolCall(name="issue_refund", args={"order_id": "123"}, id="c1")]
            ),
            ModelTurn(text="done"),
        ]
    )
    agent = acuto.Agent(model=model, instructions="test")

    @agent.tool
    def issue_refund(order_id: str) -> dict:
        "Issue a refund for an order."
        calls.append(order_id)
        return {"refunded": True}

    agent.attach(acuto.ContractLibrary("t", [acuto.action("issue_refund").at_most(2)]))

    orig_next = control.SymbolicController.next_decision
    orig_validate = control.SymbolicController.validate
    state = {"predicted": False}

    def next_decision(self, candidates, agent_state=None):
        if not state["predicted"]:
            state["predicted"] = True
            return StepDecision.symbolic(ProposedAction("issue_refund", {"order_id": "999"}))
        return orig_next(self, candidates, agent_state)

    def validate(self, action, agent_state=None):
        if action.args.get("order_id") == "999":
            return Decision(DecisionKind.BLOCK, reasons=["wrong prediction"])
        return orig_validate(self, action, agent_state)

    monkeypatch.setattr(control.SymbolicController, "next_decision", next_decision)
    monkeypatch.setattr(control.SymbolicController, "validate", validate)

    result = agent.run("Refund order #123")

    assert calls == ["123"]  # the blocked prediction (999) never executed
    assert result.status == "completed"
    assert result.tracer.by_kind("controller/fallback")
    assert model.call_count >= 1
