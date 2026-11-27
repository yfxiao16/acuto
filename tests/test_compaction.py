"""History compaction: messages are truncated or old tool results masked;
the symbolic control state is untouched and re-injected as text."""

from __future__ import annotations

import itertools
import re

import pytest

import acuto
from acuto.loop import (
    RECALL,
    ContextArchive,
    MaskCompactor,
    TruncateCompactor,
    context_tokens,
    estimate_tokens,
)
from acuto.models import MockModel, ModelTurn, ToolCall

OBLIGATION = "close the case before finishing"


def tools():
    reg = acuto.ToolRegistry()

    @reg.tool
    def work(i: int) -> dict:
        "Do one unit of work."
        return {"unit": i, "log": "x" * 200}

    @reg.tool
    def open_case() -> dict:
        "Open the case."
        return {"opened": True, "note": "the case must be closed later", "log": "y" * 200}

    @reg.tool
    def close_case() -> dict:
        "Close the case."
        return {"closed": True}

    return reg


CONTRACTS = [
    acuto.after("open_case").asserts("case_open"),
    acuto.when("case_open").obligates("close_case", deadline="eventually", desc=OBLIGATION),
]


def script(seen: list):
    """60 steps: work, open_case at step 3, close_case at step 50 only if the
    context still says it is owed, then a final answer."""

    def turn(step):
        def gen(messages, tools, system):
            seen.append(messages)
            if step == 3:
                return ModelTurn(tool_calls=[ToolCall("open_case", {})])
            if step == 50:
                owed = any("close_case" in str(m.get("content")) for m in messages)
                if owed:
                    return ModelTurn(tool_calls=[ToolCall("close_case", {})])
            if step >= 50:
                return ModelTurn(text="done")
            return ModelTurn(tool_calls=[ToolCall("work", {"i": step})])

        return gen

    return [turn(step) for step in range(52)]


def run(compactor=None):
    seen: list = []
    result = acuto.run(
        "Process 50 units.",
        model=MockModel(script(seen)),
        tools=tools(),
        contracts=CONTRACTS,
        max_steps=60,
        compactor=compactor,
    )
    return result, seen


def test_compaction_at_step_30_keeps_the_step_3_obligation():
    _, uncompacted = run()
    threshold = estimate_tokens(uncompacted[30]) - 1  # first crossed at step 30
    result, seen = run(TruncateCompactor(threshold=threshold, keep_turns=5))

    compactions = result.tracer.by_kind("context/compacted")
    assert compactions[0]["step"] == 30
    # step 3's result is gone from the context; the obligation is not
    at_50 = seen[50]
    assert not any("must be closed later" in str(m.get("content")) for m in at_50)
    note = at_50[1]["content"]
    # issued by step 3's result, so active from the next event
    assert f"- {OBLIGATION}: call `close_case` (deadline: eventually; active since event 4)" in note
    assert "- case_open = true" in note
    assert at_50[0]["content"] == "Process 50 units."
    # the model closed the case itself, from the re-injected state
    assert result.symbolic_steps == 0
    assert result.status == "completed"
    assert result.finalize["pending_obligations"] == []


def test_truncation_keeps_whole_turns():
    messages = [{"role": "user", "content": "task"}]
    for i in range(6):
        messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "r"})
    out = TruncateCompactor(threshold=0, keep_turns=2).compact(messages, "STATE")
    assert out[0] == messages[0]
    assert out[1]["role"] == "user" and out[1]["content"].endswith("STATE")
    assert out[2]["role"] == "assistant" and out[2:] == messages[-4:]
    assert TruncateCompactor(keep_turns=6).compact(messages, "STATE") == messages


def test_agent_passes_its_compactor_to_the_loop():
    _, uncompacted = run()
    threshold = estimate_tokens(uncompacted[30]) - 1
    seen: list = []
    agent = acuto.Agent(
        MockModel(script(seen)),
        tools=tools(),
        contracts=CONTRACTS,
        max_steps=60,
        compactor=TruncateCompactor(threshold=threshold, keep_turns=5),
    )
    result = agent.run("Process 50 units.")
    assert result.tracer.by_kind("context/compacted")[0]["step"] == 30


# ---------------------------------------------------------------------------
# Masking old tool results, with recall
# ---------------------------------------------------------------------------


def first_crossing_at_30(**kwargs):
    # calibrated on a run that recites the control state but never compacts
    _, uncompacted = run(MaskCompactor(threshold=10**9, min_result_tokens=0))
    return MaskCompactor(
        threshold=estimate_tokens(uncompacted[30]) - 1, keep_turns=5, min_result_tokens=0, **kwargs
    )


def stubs(messages):
    return [
        m for m in messages if m["role"] == "tool" and m["content"].startswith("[Acuto] Result")
    ]


def test_masking_keeps_every_turn_and_elides_old_results():
    result, seen = run(first_crossing_at_30())
    compactions = result.tracer.by_kind("context/compacted")
    assert compactions[0]["step"] == 30
    at_50 = seen[50]
    # every assistant turn survives: steps 0..49 each made one
    assert sum(m["role"] == "assistant" for m in at_50) == 50
    assert not any("must be closed later" in str(m.get("content")) for m in at_50)
    (opened,) = [m for m in stubs(at_50) if "open_case" in m["content"]]
    assert f'{RECALL}(id="{opened["tool_call_id"]}")' in opened["content"]
    # one control note, the latest, with the obligation
    notes = [m for m in at_50 if str(m.get("content")).startswith("[Acuto] Control state")]
    assert len(notes) == 1
    assert f"- {OBLIGATION}: call `close_case`" in notes[0]["content"]
    assert result.symbolic_steps == 0
    assert result.finalize["pending_obligations"] == []


def test_masking_is_batched_down_to_the_target():
    result, _ = run(first_crossing_at_30())
    steps = [c["step"] for c in result.tracer.by_kind("context/compacted")]
    # one batch buys many steps before the next crossing
    assert len(steps) >= 2 and all(b - a >= 5 for a, b in itertools.pairwise(steps))


def test_masking_stops_at_the_target():
    messages = [{"role": "user", "content": "task"}]
    for i in range(20):
        call = {"id": f"c{i}", "name": "t", "args": {}}
        messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        messages.append(
            {"role": "tool", "tool_call_id": f"c{i}", "name": "t", "content": "r" * 800}
        )
    before = estimate_tokens(messages)
    out = MaskCompactor(threshold=before - 1, keep_turns=2, min_result_tokens=0).compact(
        messages, "Control state", ContextArchive()
    )
    assert estimate_tokens(out) <= before * 0.5 + 250  # down to target, plus the note
    assert len(stubs(out)) < 18  # oldest first, not every eligible result


def test_recall_restores_a_masked_result_outside_the_trace():
    seen: list = []
    exposed: list = []

    def at(step, base):
        def gen(messages, tools, system):
            exposed.append([t["name"] for t in tools])
            if step == 40:
                ids = re.findall(
                    rf'{RECALL}\(id="([^"]+)"\)',
                    " ".join(
                        str(m.get("content"))
                        for m in messages
                        if "open_case" in str(m.get("content"))
                    ),
                )
                return ModelTurn(tool_calls=[ToolCall(RECALL, {"id": ids[0]})])
            return base(messages, tools, system)

        return gen

    script_ = script(seen)
    result = acuto.run(
        "Process 50 units.",
        model=MockModel([at(i, f) for i, f in enumerate(script_)]),
        tools=tools(),
        contracts=CONTRACTS,
        max_steps=60,
        compactor=first_crossing_at_30(),
    )
    assert all(RECALL in names for names in exposed)  # stable tools block
    (rec,) = result.tracer.by_kind("context/recalled")
    assert rec["ok"] and rec["step"] == 40
    # step 40 called no base script turn, so seen[40] is what step 41 saw
    assert "must be closed later" in seen[40][-1]["content"]
    # a view operation: never an event of the controller's trace
    assert RECALL not in [tool for tool, _ in result.controller.fact_history]


def test_unknown_recall_id_is_an_error_result():
    archive = ContextArchive()
    assert archive.get("nope") is None and not archive


def test_large_result_is_offloaded_on_arrival():
    reg = acuto.ToolRegistry()

    @reg.tool
    def dump() -> dict:
        "Return a lot."
        return {"rows": [f"row {i}" for i in range(5000)]}

    @reg.tool
    def other() -> dict:
        "Unused; two tools keep the model in charge."
        return {}

    def recall_it(messages, tools, system):
        cid = re.findall(rf'{RECALL}\(id="([^"]+)"\)', messages[-1]["content"])[0]
        return ModelTurn(tool_calls=[ToolCall(RECALL, {"id": cid})])

    model = MockModel(
        [ModelTurn(tool_calls=[ToolCall("dump", {})]), recall_it, ModelTurn(text="ok")]
    )
    result = acuto.run(
        "Dump.",
        model=model,
        tools=reg,
        compactor=MaskCompactor(threshold=10**9, max_result_tokens=500),
    )
    shown = model.calls[1]["messages"][-1]["content"]
    assert len(shown) < 5000 and shown.startswith('{"ok": true') and "row 4999" in shown
    page = model.calls[2]["messages"][-1]["content"]  # recalled in pages of 500 tokens
    assert page.startswith('{"ok": true') and len(page) < 2200
    assert f'{RECALL}(id="call_1", offset=2000)' in page
    assert result.status == "completed"


def test_too_small_a_batch_is_skipped():
    messages = [{"role": "user", "content": "task"}]
    for i in range(6):
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": f"c{i}", "name": "t", "args": {}}],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": f"c{i}", "name": "t", "content": "r" * 400}
        )
    archive = ContextArchive()
    lazy = MaskCompactor(threshold=10, keep_turns=2, clear_at_least=10**6, min_result_tokens=0)
    assert lazy.compact(messages, "Control state", archive) == messages and not archive
    eager = MaskCompactor(threshold=10, keep_turns=2, clear_at_least=1, min_result_tokens=0)
    out = eager.compact(messages, "Control state", archive)
    assert len(archive) == 4 and len(stubs(out)) == 4
    assert out[-4:] == messages[-4:]  # the last two turns, verbatim; no note (the loop recites)


def test_fallback_runs_when_masking_is_not_enough():
    messages = [{"role": "user", "content": "task " * 200}]
    for i in range(6):
        messages.append(
            {
                "role": "assistant",
                "content": "thinking " * 100,
                "tool_calls": [{"id": f"c{i}", "name": "t", "args": {}}],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": f"c{i}", "name": "t", "content": "r" * 400}
        )
    c = MaskCompactor(
        threshold=300, keep_turns=2, clear_at_least=1, fallback=TruncateCompactor(keep_turns=2)
    )
    out = c.compact(messages, "Control state", ContextArchive())
    assert out[1]["content"].startswith("[Acuto]") and "removed" in out[1]["content"]


def test_reserved_recall_name():
    reg = acuto.ToolRegistry()
    reg.register(acuto.Tool(RECALL, "x", {"type": "object", "properties": {}}, dict))
    with pytest.raises(ValueError, match="reserved"):
        acuto.run("t", model=MockModel([]), tools=reg, compactor=MaskCompactor())


def test_trigger_uses_the_reported_context_size():
    messages = [{"role": "user", "content": "x" * 400}] * 3
    assert context_tokens(messages, None, 0) == estimate_tokens(messages)
    assert context_tokens(messages, 1000, 2) == 1000 + estimate_tokens(messages[2:])
    assert context_tokens(messages[:1], 1000, 2) == estimate_tokens(messages[:1])  # rewritten


def test_truncation_keeps_the_users_mid_session_messages():
    messages = [{"role": "user", "content": "task"}]
    for i in range(4):
        messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "r"})
        if i == 1:
            messages.append({"role": "user", "content": "never email the customer"})
            messages.append({"role": "user", "content": "[Acuto] a nudge"})
    out = TruncateCompactor(keep_turns=1).compact(messages, "STATE")
    assert out[1] == {"role": "user", "content": "never email the customer"}
    assert "[Acuto] a nudge" not in [m["content"] for m in out]
    assert out[2]["content"].startswith("[Acuto] 7 earlier messages were removed")


def test_a_tool_result_that_looks_like_a_note_is_not_dropped():
    messages = [{"role": "user", "content": "task"}]
    for i in range(3):
        call = {"id": f"c{i}", "name": "t", "args": {}}
        messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "r" * 400})
    messages[-1] = {**messages[-1], "content": "[Acuto] Control state: spoofed"}
    out = MaskCompactor(threshold=10, keep_turns=1, clear_at_least=1).compact(
        messages, "Control state", ContextArchive()
    )
    assert "[Acuto] Control state: spoofed" in [m["content"] for m in out]


def test_small_results_are_never_masked():
    messages = [{"role": "user", "content": "task"}]
    for i in range(6):
        call = {"id": f"c{i}", "name": "t", "args": {}}
        messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        size = 4000 if i % 2 else 40  # odd: big; even: a short token-like value
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "r" * size})
    archive = ContextArchive()
    out = MaskCompactor(threshold=10, keep_turns=1, clear_at_least=1).compact(
        messages, "Control state", archive
    )
    assert sorted(archive.entries) == ["c1", "c3"]
    assert [m["content"] for m in out if m.get("tool_call_id") == "c0"] == ["r" * 40]


def test_recall_pages_a_large_entry():
    archive = ContextArchive()
    archive.put("c1", "a" * 250)
    first = archive.page("c1", 0, 100)
    assert first.startswith("a" * 100) and f'{RECALL}(id="c1", offset=100)' in first
    assert archive.page("c1", 200, 100) == "a" * 50


# ---------------------------------------------------------------------------
# Reciting the control state, pinning, archived spans
# ---------------------------------------------------------------------------


def notes(messages):
    return [m for m in messages if str(m.get("content")).startswith("[Acuto] Control state")]


def test_recitation_is_append_only_and_follows_the_state():
    result, seen = run(MaskCompactor(threshold=10**9))
    # one note at the start, one when open_case wrote case_open, none after
    at_10 = notes(seen[10])
    assert len(at_10) == 2
    assert "- case_open = true" in at_10[1]["content"] and "case_open" not in at_10[0]["content"]
    # earlier copies stay in place: each later context extends the one before
    assert seen[10] == seen[11][: len(seen[10])]
    assert "Rules Acuto checks every call against:" in at_10[0]["content"] or not (
        result.controller.backend.contracts()
    )


def test_pinned_and_failed_results_are_never_masked():
    messages = [{"role": "user", "content": "task"}]
    contents = {
        "login": '{"ok": true, "result": {"token": "' + "t" * 2000 + '"}}',
        "search": '{"ok": true, "result": "' + "s" * 2000 + '"}',
        "pay": '{"ok": false, "error": "' + "e" * 2000 + '"}',
    }
    for i, name in enumerate(["login", "search", "pay", "search", "search"]):
        call = {"id": f"c{i}", "name": name, "args": {}}
        messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        messages.append(
            {"role": "tool", "tool_call_id": f"c{i}", "name": name, "content": contents[name]}
        )
    archive = ContextArchive()
    MaskCompactor(threshold=10, keep_turns=1, clear_at_least=1, pin_tools=["login"]).compact(
        messages, "Control state", archive
    )
    assert sorted(archive.entries) == ["c1", "c3"]  # not the login, not the failure


def test_truncation_archives_the_dropped_span():
    messages = [{"role": "user", "content": "task"}]
    for i in range(4):
        messages.append(
            {"role": "assistant", "content": f"turn {i}", "tool_calls": [{"id": f"c{i}"}]}
        )
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"})
    archive = ContextArchive()
    out = TruncateCompactor(keep_turns=1).compact(messages, "STATE", archive)
    assert f'{RECALL}(id="span-1")' in out[1]["content"]
    assert "result 2" in archive.get("span-1") and "result 3" not in archive.get("span-1")


# ---------------------------------------------------------------------------
# Structured summary, the last rung
# ---------------------------------------------------------------------------


def turns(n, user_at=None, pad=0):
    messages = [{"role": "user", "content": "Pay invoice INV-2041."}]
    for i in range(n):
        call = {"id": f"c{i}", "name": "lookup", "args": {"page": i}}
        messages.append({"role": "assistant", "content": f"looking {i}", "tool_calls": [call]})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"c{i}",
                "name": "lookup",
                "content": f"row {i}: acct-77{i}" + "x" * pad,
            }
        )
        if i == user_at:
            messages.append({"role": "user", "content": "do not pay from savings"})
    return messages


def summarizer(text):
    return MockModel([ModelTurn(text=text)] * 5)


def test_summary_replaces_the_old_span_and_keeps_the_raw_tail():
    from acuto.loop import SummaryCompactor

    model = summarizer("1. Position: looked up pages 0-5.\n2. Values: account acct-773")
    c = SummaryCompactor(model, threshold=0, keep_turns=2)
    archive = ContextArchive()
    messages = turns(8, user_at=1)
    out = c.compact(messages, "Control state", archive)
    assert c.last_info == {"summary": "used"}
    assert out[0] == messages[0]
    assert out[1] == {"role": "user", "content": "do not pay from savings"}  # kept verbatim
    assert out[2]["content"].startswith("[Acuto] Summary of") and "acct-773" in out[2]["content"]
    assert out[3:] == messages[-4:]  # the raw tail
    assert "row 0: acct-770" in archive.get("span-1")
    # the summarizer saw the span, not the instruction to keep compliance out of view
    prompt = model.calls[0]["messages"][0]["content"]
    assert "row 0: acct-770" in prompt and "Do not state which actions are permitted" in prompt


def test_without_keep_user_the_users_messages_are_summarized_too():
    from acuto.loop import SummaryCompactor

    model = summarizer("1. Position: looked up pages 0-5.\n2. Values: account acct-773")
    c = SummaryCompactor(model, threshold=0, keep_turns=2, keep_user=False)
    archive = ContextArchive()
    messages = turns(8, user_at=1)
    out = c.compact(messages, "Control state", archive)
    assert out[0] == messages[0]  # the task stays
    assert all(m.get("content") != "do not pay from savings" for m in out)
    assert out[1]["content"].startswith("[Acuto] Summary of")
    assert "do not pay from savings" in model.calls[0]["messages"][0]["content"]


def test_a_summary_that_invents_a_literal_is_rejected():
    from acuto.loop import SummaryCompactor

    c = SummaryCompactor(summarizer("Values: account acct-9999"), threshold=0, keep_turns=2)
    out = c.compact(turns(8), "Control state", ContextArchive())
    assert c.last_info == {"summary": "rejected", "invented": ["acct-9999"]}
    assert "acct-9999" not in str(out) and "removed to fit" in out[1]["content"]


def test_summary_waits_for_a_boundary_below_the_hard_threshold():
    from acuto.loop import SummaryCompactor

    _, calibration = run(MaskCompactor(threshold=10**9))
    threshold = estimate_tokens(calibration[10])  # crossed from about step 10
    model = summarizer("1. Position: working through units.")
    result, _ = run(
        SummaryCompactor(model, threshold=threshold, keep_turns=3, hard_threshold=10**9)
    )
    steps = [c["step"] for c in result.tracer.by_kind("context/compacted")]
    # open_case at step 3 opens an obligation that close_case discharges at step 50:
    # nothing in between is a boundary
    assert steps and all(s > 50 or s < 4 for s in steps)


def test_masking_falls_back_to_a_summary():
    from acuto.loop import SummaryCompactor

    model = summarizer("1. Position: looked up pages.")
    c = MaskCompactor(
        threshold=200,
        keep_turns=2,
        min_result_tokens=0,
        clear_at_least=1,
        fallback=SummaryCompactor(model, threshold=200, keep_turns=2),
    )
    out = c.compact(turns(30, pad=400), "Control state", ContextArchive())
    assert c.last_info.get("summary") == "used" and c.last_info.get("masked")
    assert any(str(m.get("content")).startswith("[Acuto] Summary of") for m in out)


# ---------------------------------------------------------------------------
# Provider prompt caching
# ---------------------------------------------------------------------------


def test_anthropic_requests_cache_the_system_prompt_and_the_conversation():
    from acuto.models.anthropic import AnthropicModel

    body = AnthropicModel(api_key="x").request([{"role": "user", "content": "hi"}], [], "SYS")
    assert body["system"] == [
        {"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}
    ]
    assert body["cache_control"] == {"type": "ephemeral"}
    plain = AnthropicModel(api_key="x", cache=False).request(
        [{"role": "user", "content": "hi"}], [], "SYS"
    )
    assert plain["system"] == "SYS" and "cache_control" not in plain


def test_a_fast_forward_note_is_masked_under_an_archive_id():
    big = '{"ok": true, "result": "' + "z" * 3000 + '"}'
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "user",
            "content": f"[Acuto] Procedural step executed automatically (only move): fetch({{}}) -> {big}. Continue with the task.",
        },
    ]
    for i in range(3):
        call = {"id": f"c{i}", "name": "t", "args": {}}
        messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "name": "t", "content": "r"})
    archive = ContextArchive()
    out = MaskCompactor(threshold=10, keep_turns=1, clear_at_least=1).compact(
        messages, "Control state", archive
    )
    assert "z" * 3000 in archive.get("auto-1")
    assert (
        out[1]["content"].endswith(f'call {RECALL}(id="auto-1")]') and len(out[1]["content"]) < 300
    )
    again = MaskCompactor(threshold=10, keep_turns=1, clear_at_least=1).compact(out, "C", archive)
    assert again[1] == out[1] and len(archive) == 1  # never masked twice
