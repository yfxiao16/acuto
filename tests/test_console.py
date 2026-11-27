"""The terminal renderer: verbose runs print every decision, replay
matches, and colour is off when the stream is not a terminal."""

from __future__ import annotations

import io

import acuto
from acuto.console import Console, render
from acuto.models import MockModel, ModelTurn, ToolCall


def _run(registry, contracts, world, **kwargs):
    world["fraud"] = True
    model = MockModel(
        [
            ModelTurn(tool_calls=[ToolCall("issue_refund", {"order_id": "o1", "amount": 1.0})]),
            ModelTurn(tool_calls=[ToolCall("verify_identity", {"user_id": "u1"})]),
            ModelTurn(tool_calls=[ToolCall("check_fraud", {"user_id": "u1"})]),
            ModelTurn(tool_calls=[ToolCall("issue_refund", {"order_id": "o1", "amount": 1.0})]),
            ModelTurn(text="done"),
        ]
    )
    return acuto.run("refund", model=model, tools=registry, contracts=contracts, **kwargs)


def test_verbose_run_prints_every_decision(registry, contracts, world, capsys):
    result = _run(registry, contracts, world, verbose=True)
    out = capsys.readouterr().out
    assert "\x1b[" not in out  # captured stream is not a tty: plain text
    assert "REQUIRE" in out and "verify_identity" in out  # the premature refund
    assert "FAST-FORWARD" in out and "freeze_account" in out  # the obligation
    assert "BLOCK" in out and "forbidden while fraud_detected" in out
    assert "! obligation" in out and "✓ obligation" in out
    assert f"{result.model_calls} model calls" in out


def test_replay_matches_live_rendering(registry, contracts, world, capsys):
    result = _run(registry, contracts, world, verbose=True)
    live = capsys.readouterr().out
    buf = io.StringIO()
    render(result, stream=buf, color=False)
    replay = buf.getvalue()
    # Same step blocks and summary; replay adds a banner on top.
    assert live.strip() in replay


def test_agent_verbose_banner(registry, contracts, world, capsys):
    world["fraud"] = False
    library = acuto.ContractLibrary("refund-v1", contracts)
    library.verify()
    agent = acuto.Agent(
        model=MockModel([ModelTurn(text="nothing to do")]), tools=registry, contracts=library
    )
    agent.run("hi", verbose=True)
    out = capsys.readouterr().out
    assert "Acuto" in out and "refund-v1" in out and "verified" in out
    assert "model MockModel" in out and "control full" in out


def test_color_flag():
    buf = io.StringIO()
    Console(buf, color=True).summary(
        type(
            "R",
            (),
            {
                "finalize": {},
                "status": "completed",
                "steps": 1,
                "model_calls": 1,
                "symbolic_steps": 0,
                "symbolic_execution_ratio": 0.0,
                "blocked_proposals": 0,
                "final_text": None,
                "time_model_s": 0.0,
                "time_controller_s": 0.0,
                "time_tools_s": 0.0,
            },
        )()
    )
    assert "\x1b[" in buf.getvalue()
