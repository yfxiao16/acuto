"""Checkpoint and resume: the controller's journal replayed into a fresh
controller continues a run exactly, or under changed contracts."""

from __future__ import annotations

import pytest

import acuto
from acuto.checkpoint import Checkpoint
from acuto.loop import MaskCompactor
from acuto.models import MockModel, ModelTurn, ToolCall
from tests.test_compaction import CONTRACTS, script, tools


def go(model, max_steps, **kwargs):
    return acuto.run(
        "Process 50 units.",
        model=model,
        tools=tools(),
        contracts=kwargs.pop("contracts", CONTRACTS),
        max_steps=max_steps,
        **kwargs,
    )


def state(result):
    ctl = result.controller
    return (
        [(t, f) for t, f in ctl.fact_history],
        ctl.facts.snapshot(),
        [ob.desc for ob in ctl.obligations.pending()],
        ctl.backend.state_signature(),
        result.status,
        result.steps,
    )


def test_resume_continues_exactly_where_the_run_stopped():
    whole = go(MockModel(script([])), 60)
    first = go(MockModel(script([])), 20)
    assert first.status == "max_steps" and first.checkpoint.step == 20
    assert first.controller.obligations.pending()  # open_case at step 3, closed at 50
    rest = go(MockModel(script([])[20:]), 40, resume=first.checkpoint)
    assert state(rest) == state(whole)
    assert rest.model_calls == whole.model_calls
    assert rest.checkpoint.journal == whole.checkpoint.journal


def test_checkpoint_survives_json(tmp_path):
    first = go(MockModel(script([])), 20)
    path = tmp_path / "ckpt.json"
    first.checkpoint.save(str(path))
    loaded = Checkpoint.load(str(path))
    rest = go(MockModel(script([])[20:]), 40, resume=loaded)
    assert state(rest) == state(go(MockModel(script([])), 60))


def test_resume_keeps_the_compacted_view_and_its_archive():
    compactor = MaskCompactor(threshold=1500, keep_turns=5, min_result_tokens=0)
    first = go(MockModel(script([])), 30, compactor=compactor)
    assert first.checkpoint.archive  # results were masked before step 30
    seen: list = []
    rest = go(MockModel(script(seen)[30:]), 30, compactor=compactor, resume=first.checkpoint)
    assert rest.status == "completed"
    # the resumed context starts from the compacted view, stubs included
    assert seen[0][: len(first.checkpoint.flow["messages"])] == first.checkpoint.flow["messages"]


def test_a_swapped_contract_judges_the_recorded_trace():
    first = go(MockModel(script([])), 20)
    stricter = CONTRACTS + [acuto.action("work").at_most(5)]
    rest = go(MockModel(script([])[20:]), 40, contracts=stricter, resume=first.checkpoint)
    replayed = rest.tracer.by_kind("contract/violation")
    assert replayed and "work" in replayed[0]["tool"]
    assert "work" not in rest.controller.facts  # facts still come from the trace
    assert rest.tracer.by_kind("checkpoint/replay")[0]["step"] == 20


def test_resume_needs_a_fresh_controller_and_no_new_facts():
    first = go(MockModel(script([])), 5)
    with pytest.raises(ValueError, match="facts"):
        go(MockModel([]), 5, resume=first.checkpoint, facts={"x": True})
    used = acuto.Controller(CONTRACTS, tools=tools())
    used.assert_fact("x", True)
    with pytest.raises(ValueError, match="fresh controller"):
        go(MockModel([]), 5, controller=used, resume=first.checkpoint)


def test_journal_records_inputs_not_their_nested_writes():
    reg = tools()
    ctl = acuto.Controller(CONTRACTS, tools=reg)
    res = acuto.run(
        "t",
        model=MockModel([ModelTurn(tool_calls=[ToolCall("open_case", {})]), ModelTurn(text="x")]),
        tools=reg,
        controller=ctl,
        facts={"start": True},
        max_steps=2,
    )
    ops = [e["op"] for e in res.controller.journal]
    # the initial fact, open_case, and close_case (run by Acuto when the model
    # answers with the obligation open); case_open, written by open_case, is
    # replayed with it and has no entry of its own
    assert ops == ["assert_fact", "update", "update"]
    assert [e.get("tool") for e in res.controller.journal[1:]] == ["open_case", "close_case"]


def test_agent_resumes_from_its_last_checkpoint():
    agent = acuto.Agent(MockModel(script([])), tools=tools(), contracts=CONTRACTS, max_steps=20)
    first = agent.run("Process 50 units.")
    agent.model = MockModel(script([])[20:])
    rest = agent.run("Process 50 units.", max_steps=40, resume=first.checkpoint)
    assert state(rest) == state(go(MockModel(script([])), 60))


# ---------------------------------------------------------------------------
# Crash safety: a step is the unit of atomicity
# ---------------------------------------------------------------------------


class Crash(BaseException):
    """The process dies mid-call (not an error the tool reports)."""


def charging_tools(idempotent: bool, crash: bool):
    reg = tools()

    @reg.tool(idempotent=idempotent)
    def charge(amount: int) -> dict:
        "Charge the customer."
        if crash:
            raise Crash()
        return {"charged": amount}

    return reg


CHARGE = [ModelTurn(tool_calls=[ToolCall("work", {"i": i})]) for i in range(5)] + [
    ModelTurn(tool_calls=[ToolCall("charge", {"amount": 10})]),
    ModelTurn(text="done"),
]


def crash_at_step_5(tmp_path, idempotent):
    path = str(tmp_path / "run.json")
    with pytest.raises(Crash):
        acuto.run(
            "Charge.",
            model=MockModel(CHARGE),
            tools=charging_tools(idempotent, crash=True),
            contracts=CONTRACTS,
            checkpoint_path=path,
        )
    return path


def test_a_crash_leaves_the_steps_start_and_the_call_it_sent(tmp_path):
    ckpt = Checkpoint.load(crash_at_step_5(tmp_path, idempotent=False))
    assert ckpt.step == 5
    assert [e["tool"] for e in ckpt.journal] == ["work"] * 5  # the charge never committed
    assert ckpt.in_flight == [{"tool": "charge", "args": {"amount": 10}, "idempotent": False}]


def test_redoing_a_step_after_a_non_idempotent_call_needs_consent(tmp_path):
    ckpt = Checkpoint.load(crash_at_step_5(tmp_path, idempotent=False))
    resume = {"model": MockModel(CHARGE[5:]), "tools": charging_tools(False, crash=False)}
    with pytest.raises(acuto.InFlightError, match="charge"):
        acuto.run("Charge.", contracts=CONTRACTS, resume=ckpt, **resume)
    res = acuto.run("Charge.", contracts=CONTRACTS, resume=ckpt, on_in_flight="redo", **resume)
    assert res.status == "completed"
    assert res.tracer.by_kind("checkpoint/in_flight")[0]["calls"][0]["tool"] == "charge"
    assert [e["tool"] for e in res.controller.journal] == ["work"] * 5 + ["charge"]


def test_an_idempotent_call_is_simply_redone(tmp_path):
    ckpt = Checkpoint.load(crash_at_step_5(tmp_path, idempotent=True))
    res = acuto.run(
        "Charge.",
        model=MockModel(CHARGE[5:]),
        tools=charging_tools(True, crash=False),
        contracts=CONTRACTS,
        resume=ckpt,
    )
    assert res.status == "completed"


def test_an_agent_that_died_resumes_from_its_checkpoint_file(tmp_path):
    """The same through ``acuto.Agent``: the file written before the step
    that died holds the five calls made; the resumed agent makes only the
    one that was cut off, once the caller agrees to redo it."""
    path = str(tmp_path / "run.json")
    agent = acuto.Agent(
        MockModel(CHARGE), tools=charging_tools(False, crash=True), contracts=CONTRACTS
    )
    with pytest.raises(Crash):
        agent.run("Charge.", checkpoint_path=path)
    made = []
    reg = charging_tools(False, crash=False)
    for name in reg.names():
        tool = reg.get(name)
        tool.fn = (lambda fn, n: lambda **kw: made.append(n) or fn(**kw))(tool.fn, name)
    again = acuto.Agent(MockModel(CHARGE[5:]), tools=reg, contracts=CONTRACTS)
    with pytest.raises(acuto.InFlightError, match="charge"):
        again.run("Charge.", resume=Checkpoint.load(path))
    res = again.run("Charge.", resume=Checkpoint.load(path), on_in_flight="redo")
    assert res.status == "completed" and made == ["charge"]


def test_a_crash_safe_run_can_compact_on_the_providers_token_count(tmp_path):
    """The write-ahead list of sent calls and the count of messages the
    provider measured once shared a name: a run with a compactor, a
    checkpoint path and a model that reports its usage crashed on its
    second call."""
    turns = [
        ModelTurn(tool_calls=[ToolCall("work", {"i": i})], usage={"context": 100 + i})
        for i in range(4)
    ] + [ModelTurn(text="done")]
    res = go(
        MockModel(turns),
        10,
        contracts=[],
        compactor=MaskCompactor(threshold=10**6),
        checkpoint_path=str(tmp_path / "run.json"),
    )
    assert res.status == "completed" and res.neural_action_steps == 4


def test_changed_tool_schemas_are_refused_and_changed_contracts_traced():
    first = go(MockModel(script([])), 20)
    changed = tools()
    changed.register(acuto.Tool("extra", "x", {"type": "object", "properties": {}}, dict))
    with pytest.raises(ValueError, match="tool schemas changed"):
        acuto.run(
            "t", model=MockModel([]), tools=changed, contracts=CONTRACTS, resume=first.checkpoint
        )
    stricter = CONTRACTS + [acuto.action("work").at_most(50)]
    rest = go(MockModel(script([])[20:]), 40, contracts=stricter, resume=first.checkpoint)
    assert rest.tracer.by_kind("checkpoint/contracts_changed")


# ---------------------------------------------------------------------------
# Compaction invariance: the control state never depends on the view
# ---------------------------------------------------------------------------


def fixed_turns(recall_at=None):
    """The same 60 calls whatever the context holds; with ``recall_at``,
    an extra turn that reads an archived result back."""
    turns = []
    for step in range(60):
        if step == recall_at:
            turns.append(ModelTurn(tool_calls=[ToolCall(acuto.loop.RECALL, {"id": "call_1"})]))
        if step == 3:
            turns.append(ModelTurn(tool_calls=[ToolCall("open_case", {})]))
        elif step == 50:
            turns.append(ModelTurn(tool_calls=[ToolCall("close_case", {})]))
        else:
            turns.append(ModelTurn(tool_calls=[ToolCall("work", {"i": step})]))
    return turns + [ModelTurn(text="done")]


def test_the_control_state_is_invariant_under_every_compactor():
    from acuto.loop import SummaryCompactor, TruncateCompactor

    summarizer = MockModel([ModelTurn(text="1. Position: working.")] * 100)
    compactors = [
        None,
        TruncateCompactor(threshold=1500, keep_turns=3),
        MaskCompactor(threshold=1500, keep_turns=3, min_result_tokens=0),
        SummaryCompactor(summarizer, threshold=1500, keep_turns=3, hard_threshold=1500),
    ]
    digests, journals = [], []
    for c in compactors:
        recall = 40 if isinstance(c, MaskCompactor) else None
        res = go(MockModel(fixed_turns(recall)), 80, compactor=c)
        assert res.status == "completed"
        if c is not None:
            assert res.tracer.by_kind("context/compacted"), type(c).__name__
        digests.append(res.controller.state_digest())
        journals.append(res.controller.journal)
    assert len(set(digests)) == 1  # same trace, same control state, whatever the view
    assert all(j == journals[0] for j in journals)  # recall never entered the trace
