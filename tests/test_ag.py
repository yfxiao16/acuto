"""Assume-guarantee semantics: the environment side of a contract.

Acuto follows ContrAgent's reading. The guarantee is what the agent must
keep and it is in force from the start of the session; the assumption is
what the *environment* is required to keep, written over environment
predicates and maintained by suppressing the environment event that would
falsify it. A contract falls IDLE only when the environment breaks its
side anyway, and an IDLE contract no longer binds the agent.

The tests below pin that reading against the two shapes it is easy to
confuse it with: the fold ``A -> G`` (which discharges the guarantee
silently) and the old trigger reading (guarantee dormant until A fires).
"""

from __future__ import annotations

import json

import pytest
from contragent.formulas.formula import Atom, F, G, Implies, Not, Or, U

import acuto
from acuto.contracts import AssumptionScopeError
from acuto.decisions import ProposedAction
from acuto.tools import ToolResult


def _guarantee():
    # Every audit_log call must carry the approved fact.
    return G(Implies(Atom("called", "audit_log"), Atom("fact", "approved")))


def _assumption():
    # The environment never takes the audit channel away: a safety property
    # over an environment predicate, which is what an assumption must be.
    return G(Not(Atom("fact", "audit_channel_down")))


def _split_rule(name: str = "split"):
    return acuto.ag(_assumption(), _guarantee(), name=name)


def _unconditional_rule():
    return acuto.CustomRule(formula=_guarantee(), name="unconditional")


def _folded_rule():
    return acuto.CustomRule(formula=Implies(_assumption(), _guarantee()), name="folded")


def _controller(*contracts):
    return acuto.SymbolicController(list(contracts), tracer=acuto.Tracer())


def _commit(controller, tool, agent_state=None):
    action = ProposedAction(tool, {})
    controller.update(action, ToolResult(tool=tool, args={}, ok=True), agent_state)


def _violations(controller):
    return [r for r in controller.tracer.by_kind("contract/violation")]


# ---------------------------------------------------------------------------
# The guarantee binds from the start
# ---------------------------------------------------------------------------


def test_guarantee_is_in_force_from_the_start():
    """No trigger to wait for: an A/G contract constrains the agent from
    event one, exactly like the unconditional contract it degenerates to
    while the environment keeps its side."""
    for contract in (_split_rule(), _unconditional_rule()):
        c = _controller(contract)
        _commit(c, "audit_log")  # approved is not held -> guarantee broken
        assert _violations(c), contract.name


def test_masking_treats_the_two_forms_alike():
    candidates = ["audit_log", "other_tool"]
    for contract in (_split_rule(), _unconditional_rule()):
        c = _controller(contract)
        # audit_log would break the guarantee now, so the mask removes it.
        assert set(c.admissible_actions(candidates)) == {"other_tool"}
        c.assert_fact("approved")
        assert set(c.admissible_actions(candidates)) == set(candidates)


# ---------------------------------------------------------------------------
# Suppression: Acuto maintains the assumption rather than watching it fail
# ---------------------------------------------------------------------------


def test_fact_write_that_would_break_the_assumption_is_suppressed():
    c = _controller(_split_rule())
    accepted = c.assert_fact("audit_channel_down")
    assert accepted is False
    assert "audit_channel_down" not in c.facts
    suppressed = c.tracer.by_kind("fact/suppressed")
    assert len(suppressed) == 1 and suppressed[0]["contracts"] == ["split"]
    # The contract is untouched, so the guarantee still binds.
    assert c.finalize().get("idle_contracts") is None
    _commit(c, "audit_log")
    assert _violations(c)


def test_extractor_writes_go_through_the_same_gate():
    """A fact asserted from a tool result is an environment event, so it is
    suppressed on the same rule as a direct write, at the level of the
    return event that carries it."""
    rules = [
        _split_rule(),
        acuto.after("probe_channel").asserts("audit_channel_down"),
    ]
    c = _controller(*rules)
    _commit(c, "probe_channel")
    assert "audit_channel_down" not in c.facts
    assert c.tracer.by_kind("return/suppressed")


def test_unrelated_fact_writes_are_untouched():
    c = _controller(_split_rule())
    assert c.assert_fact("approved") is True
    assert c.facts.truthy("approved")
    assert not c.tracer.by_kind("fact/suppressed")


def test_suppression_costs_nothing_without_assumptions():
    c = _controller(_unconditional_rule())
    assert c.contracts.has_assumptions is False
    assert c.assert_fact("anything") is True


# ---------------------------------------------------------------------------
# IDLE: what happens if the environment breaks its side anyway
# ---------------------------------------------------------------------------


def test_broken_assumption_makes_the_contract_idle():
    """Written straight into the fact store, bypassing the suppression gate:
    the contract goes IDLE and stops constraining the agent, and finalize
    attributes that to the environment."""
    c = _controller(_split_rule())
    c.facts.assert_fact("audit_channel_down")  # bypasses assert_fact
    _commit(c, "audit_log")  # would break the guarantee...
    assert not _violations(c)  # ...but the contract is IDLE
    report = c.finalize()
    assert report["idle_contracts"] == ["split"]
    assert report["ltlf_violations"] == []
    assert c.tracer.by_kind("contract/assumption_broken")


def test_split_form_names_the_environment_where_the_fold_stays_silent():
    """The point of keeping A and G apart: the fold A -> G is satisfied by a
    broken assumption and reports nothing, while the split form records an
    IDLE contract naming the environment as the cause."""
    split, folded = _controller(_split_rule()), _controller(_folded_rule())
    for c in (split, folded):
        c.facts.assert_fact("audit_channel_down")
        _commit(c, "audit_log")
        assert not _violations(c)
    assert split.finalize()["idle_contracts"] == ["split"]
    assert "idle_contracts" not in folded.finalize()


# ---------------------------------------------------------------------------
# What an assumption may say
# ---------------------------------------------------------------------------


def test_assumption_over_agent_predicates_is_rejected():
    """A trigger on the agent's own calls is not the environment's to keep."""
    with pytest.raises(AssumptionScopeError) as exc:
        acuto.ContractLibrary(
            "bad", [acuto.ag(F(Atom("called", "enable_audit")), _guarantee())]
        ).compiled
    assert "called" in str(exc.value)
    assert "G(trigger -> ...)" in str(exc.value)


def test_agent_side_trigger_belongs_in_the_guarantee():
    """The migration the error message prescribes, and it behaves."""
    rule = acuto.CustomRule(
        formula=G(Implies(Atom("called", "audit_log"), Atom("fact", "approved"))),
        name="migrated",
    )
    c = _controller(rule)
    _commit(c, "other_tool")
    assert not _violations(c)
    _commit(c, "audit_log")
    assert _violations(c)


def test_assumption_demanding_an_eventuality_is_rejected():
    """Suppression can withhold an event, never bring one about."""
    with pytest.raises(AssumptionScopeError) as exc:
        acuto.ContractLibrary(
            "bad", [acuto.ag(F(Atom("fact", "channel_restored")), _guarantee())]
        ).compiled
    assert "eventuality" in str(exc.value)


def test_weak_until_assumption_is_accepted():
    """(phi U psi) | G phi is a safety property and stays allowed."""
    phi, psi = Atom("fact", "provisional"), Atom("fact", "confirmed")
    weak_until = Or(U(phi, psi), G(phi))
    lib = acuto.ContractLibrary("wu", [acuto.ag(weak_until, _guarantee())])
    assert len(lib.contracts) == 1


# ---------------------------------------------------------------------------
# Certificates and import/export
# ---------------------------------------------------------------------------


def test_verify_reports_compatibility():
    lib = acuto.ContractLibrary("ag-lib", [_split_rule()])
    report = lib.verify()
    assert report.ok
    assert report.contracts[0].assumption_satisfiable is True


def test_joint_query_keeps_every_assumption():
    """The library query is /\\(A_i /\\ G_i): two contracts whose assumptions
    cannot hold together are a conflict, even though each fold A_i -> G_i is
    satisfiable on its own."""
    up = Atom("fact", "channel_up")
    a1, a2 = G(up), G(Not(up))
    lib = acuto.ContractLibrary(
        "clash",
        [acuto.ag(a1, _guarantee(), name="c1"), acuto.ag(a2, _guarantee(), name="c2")],
    )
    report = lib.verify(strict=False)
    assert report.jointly_satisfiable is False


def test_from_contragent_contract_objects():
    from contragent.models.agent import Agent as CAgent
    from contragent.models.contract import Contract

    contract = Contract(
        agent=CAgent(id="assistant", tools=[]),
        assumption=_assumption(),
        guarantee=_guarantee(),
        desc="audit discipline",
    )
    lib = acuto.ContractLibrary.from_contragent([contract], name="imported")
    assert len(lib.contracts) == 1
    imported = lib.contracts[0]
    assert imported.assumption is not None and imported.name == "audit discipline"
    # Behaves per A/G semantics end to end.
    c = acuto.SymbolicController(lib.compiled, tracer=acuto.Tracer())
    assert c.assert_fact("audit_channel_down") is False
    _commit(c, "audit_log")
    assert _violations(c)


def test_round_trip_through_contragent():
    """to_contragent() must produce contracts ContrAgent itself accepts —
    the scope check runs on its side too."""
    lib = acuto.ContractLibrary("rt", [_split_rule()])
    exported = lib.to_contragent()
    assert len(exported) == 1 and exported[0].assumption is not None


def test_from_contragent_yaml(tmp_path):
    yaml = tmp_path / "bank.yaml"
    yaml.write_text(
        """\
version: "1"
agents:
  "*":
    contracts:
      - desc: "Hold transfers"
        G:
          ltl: "G(!called(transfer))"
"""
    )
    lib = acuto.ContractLibrary.from_contragent(yaml, name="bank")
    assert len(lib.contracts) == 1
    # Every imported contract compiles into a live monitor.
    from acuto.backend import LTLfBackend

    LTLfBackend(lib.contracts)


def test_deep_conflict_check_via_contragent():
    lib = acuto.ContractLibrary("ag-lib", [_split_rule()])
    report = lib.check_conflicts()
    assert report.ok, report.render()


def test_soft_tier_contracts_do_not_mask_but_still_validate():
    from acuto.contracts import CustomRule
    from acuto.control import SymbolicController

    # Soft precedence: never call b before a — mined convention, masking=False.
    soft = CustomRule(
        formula=Or(U(Not(Atom("called", "b")), Atom("called", "a")), G(Not(Atom("called", "b")))),
        name="soft a-before-b",
        masking=False,
    )
    lib = acuto.ContractLibrary("soft-tier", [soft])
    ctl = SymbolicController(lib)
    # b stays exposed (no silent removal)...
    assert "b" in ctl.admissible_actions(["a", "b"])
    # ...but validating a concrete premature call still reports the violation.
    decision = ctl.validate(ProposedAction(tool="b", args={}))
    assert decision.kind.name != "ALLOW"
    assert any("soft a-before-b" in r for r in decision.reasons)


# ---------------------------------------------------------------------------
# Suppression is an enforcement action, not an observation
# ---------------------------------------------------------------------------


def test_auditor_records_the_break_instead_of_suppressing_it():
    """An auditor replays ANY condition's run, including one whose environment
    broke an assumption. Suppressing there would rewrite history and make the
    audited condition look cleaner than it was, so observe mode records the
    break and lets the write land."""
    lib = acuto.ContractLibrary("ag-lib", [_split_rule()])
    aud = lib.auditor()
    assert aud.enforcing is False
    assert aud.assert_fact("audit_channel_down") is True
    assert "audit_channel_down" in aud.facts
    assert not aud.tracer.by_kind("fact/suppressed")
    broke = aud.tracer.by_kind("fact/broke_assumption")
    assert len(broke) == 1 and broke[0]["contracts"] == ["split"]
    # ...and the break still reaches the end-of-session report.
    _commit(aud, "other_tool")
    assert aud.finalize()["idle_contracts"] == ["split"]


def test_enforcing_controller_still_suppresses():
    """The same library on the enforcing side keeps the guarantee in force."""
    c = _controller(_split_rule())
    assert c.enforcing is True
    assert c.assert_fact("audit_channel_down") is False
    _commit(c, "audit_log")
    assert _violations(c)


def test_audit_and_enforce_agree_when_the_environment_behaves():
    """The common case: no assumption is ever threatened, so the two modes are
    indistinguishable — which is what the condition-independent audit needs."""
    lib = acuto.ContractLibrary("ag-lib", [_split_rule()])
    enf, aud = acuto.SymbolicController(lib.compiled, tracer=acuto.Tracer()), lib.auditor()
    for c in (enf, aud):
        c.assert_fact("approved")
        _commit(c, "audit_log")
    assert enf.finalize() == aud.finalize()


# ---------------------------------------------------------------------------
# The return event is verified as a whole before the model sees it
# ---------------------------------------------------------------------------


def _probe_result():
    return ToolResult(tool="probe_channel", args={}, ok=True, output={"status": "down"})


def test_breaking_return_is_suppressed_as_a_whole():
    """One extractor writes a harmless fact and another the breaking one.
    The verdict is on the return event, so neither lands, the call itself
    stays committed, and the guarantee still binds afterwards."""
    rules = [
        _split_rule(),
        acuto.after("probe_channel").asserts("approved").asserts("audit_channel_down"),
    ]
    c = _controller(*rules)
    action = ProposedAction("probe_channel", {})
    verdict = c.verify_return(action, _probe_result())
    assert verdict.suppressed and verdict.breakers == ["split"]
    c.update(action, _probe_result(), return_verdict=verdict)
    assert "approved" not in c.facts and "audit_channel_down" not in c.facts
    assert [e.tool for e in c.events] == ["probe_channel"]
    assert c.tracer.by_kind("return/suppressed")
    assert c.finalize().get("idle_contracts") is None
    _commit(c, "audit_log")
    assert _violations(c)


def test_return_verdict_is_joint_not_per_write():
    """An assumption that relates two facts of one result: written one at a
    time the first write would break it, written together they satisfy it.
    The verdict is on the whole return, so both land."""
    assumption = G(Implies(Atom("fact", "channel_switched"), Atom("fact", "channel_ready")))
    rules = [
        acuto.ag(assumption, _guarantee(), name="paired"),
        acuto.after("probe_channel").asserts("channel_switched").asserts("channel_ready"),
    ]
    c = _controller(*rules)
    action = ProposedAction("probe_channel", {})
    assert not c.verify_return(action, _probe_result()).suppressed
    c.update(action, _probe_result())
    assert c.facts.truthy("channel_switched") and c.facts.truthy("channel_ready")
    assert not c.tracer.by_kind("fact/suppressed")


def test_update_without_a_verdict_computes_it():
    """Callers that only call update() get the same return-level semantics."""
    rules = [_split_rule(), acuto.after("probe_channel").asserts("audit_channel_down")]
    c = _controller(*rules)
    c.update(ProposedAction("probe_channel", {}), _probe_result())
    assert "audit_channel_down" not in c.facts
    assert c.tracer.by_kind("return/suppressed") and not c.tracer.by_kind("fact/suppressed")


def test_auditor_reports_but_does_not_suppress_a_breaking_return():
    lib = acuto.ContractLibrary(
        "ag-lib", [_split_rule(), acuto.after("probe_channel").asserts("audit_channel_down")]
    )
    aud = lib.auditor()
    action = ProposedAction("probe_channel", {})
    verdict = aud.verify_return(action, _probe_result())
    assert verdict.breakers == ["split"] and not verdict.suppressed
    aud.update(action, _probe_result(), return_verdict=verdict)
    assert "audit_channel_down" in aud.facts
    assert aud.tracer.by_kind("fact/broke_assumption")
    assert not aud.tracer.by_kind("return/suppressed")


def test_loop_withholds_a_suppressed_result_from_the_model():
    """End to end: the tool returns a result that would break the assumption.
    The model receives a withheld-result error in its place, the fact is not
    recorded, and the call itself stays in the trace."""
    from acuto.models import MockModel, ModelTurn, ToolCall
    from acuto.tools import ToolRegistry

    registry = ToolRegistry()

    def probe_channel():
        return {"status": "down"}

    def audit_log():
        return {"logged": True}

    schema = {"type": "object", "properties": {}}
    registry.tool(probe_channel, name="probe_channel", description="probe", parameters=schema)
    registry.tool(audit_log, name="audit_log", description="log", parameters=schema)
    lib = acuto.ContractLibrary(
        "ag-lib", [_split_rule(), acuto.after("probe_channel").asserts("audit_channel_down")]
    )
    model = MockModel(
        [
            ModelTurn(tool_calls=[ToolCall("probe_channel", {}, "c1")]),
            ModelTurn(text="done"),
        ]
    )
    agent = acuto.Agent(model, tools=registry, contracts=lib, control_mode="passive")
    run = agent.run("probe the channel")
    tool_msgs = [m for m in run.flow.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["ok"] is False and "split" in payload["error"]
    assert "audit_channel_down" not in run.controller.facts
    assert [e.tool for e in run.controller.events] == ["probe_channel"]
    assert run.tracer.by_kind("return/withheld")
