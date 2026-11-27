"""A call replayed from another agent's trace derives its context too."""

from __future__ import annotations

from contragent.formulas.formula import Atom, G, Implies

import acuto


def test_observe_call_derives_the_context_of_the_observed_call():
    rule = G(Implies(Atom("called", "remind"), Atom("ctx", "target_ok", "yes")))

    def derive(action, facts):
        if action.tool == "remind":
            return {"target_ok": "yes" if action.args.get("id") == 1 else "no"}
        return {}

    ctl = acuto.SymbolicController(
        [acuto.CustomRule(rule, name="remind only the right one")],
        context_deriver=derive,
        enforcing=False,
    )
    assert ctl.observe_call("remind", args={"id": 1}) == []
    broken = ctl.observe_call("remind", args={"id": 2})
    assert [c.name for c in broken] == ["remind only the right one"]


def test_a_rule_over_state_context_masks_by_the_derived_value():
    # "complete only when all is done": the mask derives all_done (it
    # depends on the state, not the arguments) instead of reading it unset.
    done = "supervisor__complete_task"
    rule = G(Implies(Atom("called", done), Atom("ctx", "all_done", "yes")))
    state = {"all_done": "no"}
    ctl = acuto.SymbolicController(
        [acuto.CustomRule(rule, name="complete only when done")],
        context_deriver=lambda a, f: dict(state) if a.tool == done else {},
    )
    assert ctl.admissible_actions([done, "delegate"]) == ["delegate"]
    state["all_done"] = "yes"
    assert ctl.admissible_actions([done, "delegate"]) == [done, "delegate"]
