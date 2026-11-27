"""Per-value contracts (ContrAgent's ForEach) and argument terms in Acuto."""

from __future__ import annotations

import pytest
import acuto

from .helpers import registry

formula = pytest.importorskip("contragent.formulas.formula")
if not hasattr(formula, "ForEach"):
    pytest.skip("this ContrAgent has no per-value relations", allow_module_level=True)


def compile_per(relation, subject, obj=None, quantifier=None, field="order_id"):
    from contragent.generation.structured_ir import ConstraintIR, compile_ir

    res = compile_ir(
        ConstraintIR(
            subject=subject,
            object=obj,
            relation=relation,
            scope="global",
            quantifier=quantifier,
            params={"per": field},
        )
    )
    assert not res.error, res.error
    return res.compiled.formula


def kind(ctl, tool, **args):
    return ctl.validate(acuto.ProposedAction(tool, args)).kind


def commit(ctl, tool, **args):
    ctl.update(
        acuto.ProposedAction(tool, args), acuto.ToolResult(tool=tool, args=args, ok=True, output={})
    )


def test_look_up_the_same_order_before_refunding_it():
    ctl = acuto.Controller([], tools=registry()).symbolic
    f = compile_per("precedes", "lookup_order", "issue_refund")
    ctl.attach_rule("r1", f, name="look up the order before refunding it", advisory=False)
    commit(ctl, "lookup_order", order_id="7")
    assert kind(ctl, "issue_refund", order_id="7") == acuto.DecisionKind.ALLOW
    assert kind(ctl, "issue_refund", order_id="8") != acuto.DecisionKind.ALLOW
    # Argument-dependent: judged at validate, never masked by name.
    assert "issue_refund" in ctl.admissible_actions(["issue_refund", "lookup_order"])


def test_refund_each_order_at_most_once():
    ctl = acuto.Controller([], tools=registry()).symbolic
    ctl.attach_rule(
        "r1",
        compile_per("limits", "issue_refund", quantifier=1),
        name="once per order",
        advisory=False,
    )
    commit(ctl, "issue_refund", order_id="7")
    assert kind(ctl, "issue_refund", order_id="8") == acuto.DecisionKind.ALLOW
    assert kind(ctl, "issue_refund", order_id="7") != acuto.DecisionKind.ALLOW


def test_a_per_value_rule_survives_a_journal_replay():
    ctl = acuto.Controller([], tools=registry()).symbolic
    ctl.attach_rule(
        "r1",
        compile_per("limits", "issue_refund", quantifier=1),
        name="once per order",
        advisory=False,
    )
    commit(ctl, "issue_refund", order_id="7")
    fresh = acuto.Controller([], tools=registry()).symbolic
    fresh.replay(ctl.journal)
    assert kind(fresh, "issue_refund", order_id="7") != acuto.DecisionKind.ALLOW


def test_complete_is_blocked_while_an_app_is_still_logged_in():
    from contragent.generation.structured_ir import ConstraintIR, compile_ir

    if not hasattr(formula.ForEach, "route"):
        pytest.skip("this ContrAgent has no tool families")
    done = "supervisor__complete_task"
    res = compile_ir(
        ConstraintIR(
            subject="*__login",
            object="*__logout",
            relation="follows",
            scope="global",
            params={"per": "@app", "by": done},
        )
    )
    assert not res.error, res.error
    ctl = acuto.SymbolicController([acuto.CustomRule(res.compiled.formula, name="log out")])

    def allowed(tool):
        return ctl.validate(acuto.ProposedAction(tool, {})).kind == acuto.DecisionKind.ALLOW

    for tool in ("spotify__login", "venmo__login", "venmo__logout"):
        assert allowed(tool)
        ctl.observe_call(tool)
    assert not allowed(done)  # spotify is still logged in
    ctl.observe_call("spotify__logout")
    assert allowed(done)
