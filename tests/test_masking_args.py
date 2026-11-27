"""Contracts that read arguments are judged at validate, never masked by name."""

from __future__ import annotations

from contragent.formulas.formula import ArgValue, Atom, Const, G, Implies, Le

import acuto

from .helpers import registry


def kind(ctl, tool, **args):
    return ctl.validate(acuto.ProposedAction(tool, args)).kind


def test_a_raw_argument_term_is_judged_at_validate_not_masked():
    # Found by the predicate inventory: ArgValue was not seen as argument-
    # dependent, so the mask probed it with no arguments and hid the tool.
    cap = G(
        Implies(Atom("called", "issue_refund"), Le(ArgValue("issue_refund", "amount"), Const(100)))
    )
    ctl = acuto.Controller([acuto.CustomRule(cap, name="cap")], tools=registry()).symbolic
    assert "issue_refund" in ctl.admissible_actions(["issue_refund", "lookup_order"])
    assert kind(ctl, "issue_refund", order_id="7", amount=50) == acuto.DecisionKind.ALLOW
    assert kind(ctl, "issue_refund", order_id="7", amount=500) != acuto.DecisionKind.ALLOW
