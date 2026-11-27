"""A guarantee ``F fact(p)`` met by the run's last call is reported met:
the facts a result writes are visible from the next event, and the end of
the run is that event (E1: the IT engineer's last call opened the incident,
yet the final report said the incident was never opened)."""

from __future__ import annotations

from contragent.formulas.formula import Atom, F

import acuto
from acuto.models import MockModel, ModelTurn, ToolCall


def test_a_guarantee_met_by_the_last_call_is_not_reported_broken():
    reg = acuto.ToolRegistry()
    reg.register(
        acuto.Tool(
            "open_incident",
            "Open it.",
            {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
            lambda title: {"number": "INC-1"},
        )
    )
    rules = [
        acuto.after("open_incident").asserts("opened"),
        acuto.CustomRule(F(Atom("fact", "opened")), name="open the incident"),
    ]
    script = [
        ModelTurn(tool_calls=[ToolCall("open_incident", {"title": "x"})]),
        ModelTurn(text="ok"),
    ]
    res = acuto.run(
        "Open the incident.",
        model=MockModel(script),
        tools=reg,
        contracts=acuto.ContractLibrary("it", rules),
        max_steps=4,
    )
    assert res.status == "completed"
    assert res.finalize["ltlf_violations"] == []
