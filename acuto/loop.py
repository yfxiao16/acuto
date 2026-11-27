"""The Acuto runtime loop. Deliberately minimal: no
procedural semantics live here — they live in the SymbolicController.

    read flow state → candidate actions → admissible actions
    → who decides? (symbolic / neural / dead end)
    → hard validation immediately before execution
    → execute → update both worlds → advance flow → trace → repeat

The core is a generator (:func:`iterate`) that yields one
:class:`Frame` per step — the streaming decision-frame surface. ``run``
just drains it.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Generator, Protocol, runtime_checkable

from acuto.checkpoint import (
    Checkpoint,
    InFlightError,
    check_stamps,
    contract_stamp,
    tool_stamp,
)
from acuto.control import SymbolicController
from acuto.controller import Controller
from acuto.decisions import DecisionKind, ProposedAction, StepDecision, StepKind
from acuto.flow import Flow, FlowContext, FreeFlow
from acuto.models.base import Model, ToolCall
from acuto.tools import ToolRegistry
from acuto.tracing import Tracer

# ---------------------------------------------------------------------------
# History compaction (docs/TODO.md A1)
# ---------------------------------------------------------------------------

RECALL = "acuto_recall"  # the loop's own tool: show an elided result again
_NOTE = "[Acuto] Control state"  # prefix of a re-injected control-state note
_AUTO = "[Acuto] Procedural step executed automatically"  # a fast-forward's result


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token count of a message list: serialized length / 4."""
    return sum(len(json.dumps(m, default=str)) for m in messages) // 4


def context_tokens(messages: list[dict], measured: int | None, sent: int) -> int:
    """Tokens the next request will carry: what the provider reported for
    the last request (``measured``, over its first ``sent`` messages) plus
    an estimate of the messages appended since. Falls back to
    :func:`estimate_tokens` before the first report or after a rewrite."""
    if measured is None or sent > len(messages):
        return estimate_tokens(messages)
    return measured + estimate_tokens(messages[sent:])


def control_state_text(controller: SymbolicController, admissible: list[str] | None = None) -> str:
    """The symbolic control state as text for the model: pending
    obligations, established facts, the rules every call is checked
    against, and (given ``admissible``) the tools it may call now. A long
    fact value is shortened; the fact itself stays in the controller."""
    lines = ["Control state (maintained by Acuto; authoritative over anything above):"]
    lines.append("Pending obligations:")
    pending = controller.obligations.pending()
    for ob in pending:
        lines.append(
            f"- {ob.desc}: call `{ob.tool}` (deadline: {ob.spec.deadline}; "
            f"active since event {ob.triggered_at})"
        )
    if not pending:
        lines.append("- (none)")
    lines.append("Established facts:")
    facts = controller.facts.snapshot()
    lines += [f"- {p} = {_short(json.dumps(v, default=str))}" for p, v in facts.items()] or [
        "- (none)"
    ]
    rules = list(
        dict.fromkeys(
            c.name + (" [proposed, not yet confirmed: warns, does not block]" if c.advisory else "")
            for c in controller.backend.contracts()
        )
    )
    if rules:
        lines.append("Rules Acuto checks every call against:")
        lines += [f"- {r}" for r in rules[:_RULES_SHOWN]]
        if len(rules) > _RULES_SHOWN:
            lines.append(f"- (+{len(rules) - _RULES_SHOWN} more)")
    if admissible is not None:
        lines.append("Tools you may call now: " + (", ".join(admissible) or "(none)"))
    return "\n".join(lines)


def _short(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


_RULES_SHOWN = 30


class ContextArchive:
    """Full text of what a compactor took out of the context: a tool result
    by its tool-call id, a dropped or summarized span of turns by a
    ``span-N`` id. The model gets an entry back with the ``acuto_recall``
    tool, so compacted content is out of view, never lost: the history is
    the view plus the archive. Recalling is not an event of the
    controller's trace and no contract sees it."""

    def __init__(self, entries: dict[str, str] | None = None) -> None:
        self.entries: dict[str, str] = dict(entries or {})

    def put(self, entry_id: str, content: str) -> None:
        self.entries[entry_id] = content

    def get(self, entry_id: str) -> str | None:
        return self.entries.get(entry_id)

    def new_span(self, messages: list[dict]) -> str:
        """Archive a span of messages; return its id."""
        span_id = f"span-{sum(k.startswith('span-') for k in self.entries) + 1}"
        self.put(span_id, json.dumps(messages, default=str))
        return span_id

    def page(self, entry_id: str, offset: int = 0, limit: int = 80_000) -> str | None:
        """``limit`` characters of an entry from ``offset``, with the
        offset of the next page when the entry goes on."""
        content = self.entries.get(entry_id)
        if content is None:
            return None
        chunk = content[offset : offset + limit]
        if offset + limit < len(content):
            chunk += (
                f"\n[Acuto] {len(content) - offset - limit} more characters: call "
                f'{RECALL}(id="{entry_id}", offset={offset + limit}).'
            )
        return chunk

    def __contains__(self, entry_id: object) -> bool:
        return entry_id in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def schema() -> dict:
        return {
            "name": RECALL,
            "description": (
                "Show again content that Acuto elided from the conversation to save "
                "context: a tool result or a span of earlier turns. Pass the id given "
                "in the elision note."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "id from the note"},
                    "offset": {"type": "integer", "description": "character offset of a page"},
                },
                "required": ["id"],
            },
        }


@runtime_checkable
class Compactor(Protocol):
    """Shrinks the flow's message list once the context outgrows
    ``threshold`` tokens (:func:`context_tokens`). :func:`iterate` calls
    it before the model call. It rewrites messages only; the symbolic
    control state stays in the controller, and ``control`` is that state
    rendered as text (:func:`control_state_text`) for the compactor to
    re-inject. Content it takes out goes to ``archive``. Returning
    ``messages`` unchanged means nothing was worth compacting.

    Optional attributes the loop reads: ``recite`` (the loop appends the
    control state whenever it changes, so the compactor need not inject
    it), ``hard_threshold`` (between the two thresholds the loop compacts
    only at a boundary, when no obligation is pending), ``admit`` (a
    method that shapes each fresh tool result) and ``last_info`` (a dict
    merged into the ``context/compacted`` trace record)."""

    threshold: int

    def compact(
        self, messages: list[dict], control: str, archive: ContextArchive
    ) -> list[dict]: ...


def _turn_starts(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if i > 0 and m["role"] == "assistant"]


def _is_note(m: dict) -> bool:
    """A control-state note Acuto itself injected (a tool result that
    happens to start the same way is not one)."""
    return m["role"] == "user" and str(m.get("content") or "").startswith(_NOTE)


def _said_by_user(m: dict) -> bool:
    """A user turn that is not one of Acuto's own notes or nudges: the
    user's mid-session instructions, which compaction must not drop."""
    return m["role"] == "user" and not str(m.get("content") or "").startswith("[Acuto]")


class TruncateCompactor:
    """Keep the task (the first message) and the last ``keep_turns``
    assistant turns with the messages after them; replace everything in
    between with one note carrying the control state. The user's own
    messages in the dropped span are kept, in order: constraints stated
    mid-session are what truncation loses first (arXiv:2608.11242). The
    dropped span goes to the archive, so the note names a recall id. No
    summary of the dropped turns is written, so nothing the model must
    comply with is left to its own recollection."""

    def __init__(self, threshold: int = 100_000, keep_turns: int = 4) -> None:
        self.threshold = threshold
        self.keep_turns = keep_turns

    def compact(
        self, messages: list[dict], control: str, archive: ContextArchive | None = None
    ) -> list[dict]:
        starts = _turn_starts(messages)
        if len(starts) <= self.keep_turns:
            return list(messages)
        cut = starts[-self.keep_turns]
        kept = [m for m in messages[1:cut] if _said_by_user(m)]
        dropped = cut - 1 - len(kept)
        where = ""
        if archive is not None:
            span = archive.new_span([m for m in messages[1:cut] if not _said_by_user(m)])
            where = f'; call {RECALL}(id="{span}") to read them'
        note = {
            "role": "user",
            "content": (
                f"[Acuto] {dropped} earlier messages were removed to fit the context "
                f"window{where}.\n{control}"
            ),
        }
        return messages[:1] + kept + [note] + messages[cut:]


class MaskCompactor:
    """Elide old tool results and keep everything else verbatim: the task,
    every assistant turn (the model's own calls and their arguments), and
    the results of the last ``keep_turns`` turns. An elided result becomes
    a one-line stub naming the call and its recall id; its text goes to
    the archive. So does the note reporting a call Acuto executed itself
    (a fast-forward), under an ``auto-N`` id. Never masked: a result under ``min_result_tokens``
    (masking it saves almost nothing and can hide a short value the agent
    still uses, an access token or an id), a failed or blocked call's
    result, and the results of ``pin_tools`` (say, a login whose token
    lives only in its result).

    Masking runs in one batch per threshold crossing, oldest results
    first, down to ``target`` x ``threshold``, so the cached prompt prefix
    is rewritten rarely; a batch that would free less than
    ``clear_at_least`` tokens (default a tenth of the threshold) is not
    worth the cache miss and is skipped. When masking cannot get under the
    threshold, ``fallback`` (:class:`TruncateCompactor` or
    :class:`SummaryCompactor`) runs on the masked history. Earlier
    control-state notes are dropped; with ``recite`` the loop appends the
    current one, otherwise the compactor does.

    ``max_result_tokens`` offloads a large result as it arrives
    (:meth:`admit`): the model sees its head and tail and the recall id.
    A recalled result comes back in pages of the same size.

    The 10-turn default window follows arXiv:2508.21433. Set the
    threshold per benchmark: AppWorld contexts peak near 8-18K tokens
    (arXiv:2510.00615), far below the default."""

    def __init__(
        self,
        threshold: int = 100_000,
        keep_turns: int = 10,
        *,
        target: float = 0.5,
        min_result_tokens: int = 250,
        clear_at_least: int | None = None,
        max_result_tokens: int | None = 20_000,
        pin_tools: Iterable[str] = (),
        recite: bool = True,
        fallback: Compactor | None = None,
    ) -> None:
        if keep_turns < 1:
            raise ValueError("keep_turns must be at least 1")
        self.threshold = threshold
        self.keep_turns = keep_turns
        self.target = target
        self.min_result_tokens = min_result_tokens
        self.clear_at_least = threshold // 10 if clear_at_least is None else clear_at_least
        self.max_result_tokens = max_result_tokens
        self.pin_tools = set(pin_tools)
        self.recite = recite
        self.fallback = fallback
        self.last_info: dict = {}

    @staticmethod
    def stub(call_id: str, name: str, args: dict | None) -> str:
        shown = _short(json.dumps(args or {}, default=str), 120)
        return (
            f"[Acuto] Result of {name}({shown}) elided to save context. "
            f'Call {RECALL}(id="{call_id}") to see it again.'
        )

    def admit(self, call_id: str, name: str, args: dict | None, content: str, archive) -> str:
        """What the model sees of a fresh result: all of it, or for one over
        ``max_result_tokens`` its head and tail around the recall stub."""
        if self.max_result_tokens is None or len(content) // 4 <= self.max_result_tokens:
            return content
        archive.put(call_id, content)
        keep = self.max_result_tokens * 2  # chars of head and of tail
        return f"{content[:keep]}\n…\n{content[-keep:]}\n{self.stub(call_id, name, args)}"

    def _maskable(self, m: dict, name: str, archive: ContextArchive) -> bool:
        cid = m.get("tool_call_id")
        content = str(m.get("content") or "")
        if m["role"] == "user" and content.startswith(_AUTO):
            # A call Acuto executed itself reaches the model as this note,
            # result included; it has no call id, so the archive names it.
            return RECALL not in content and estimate_tokens([m]) >= self.min_result_tokens
        return (
            m["role"] == "tool"
            and cid is not None
            and cid not in archive
            and name not in self.pin_tools
            and not content.startswith('{"ok": false')
            and estimate_tokens([m]) >= self.min_result_tokens
        )

    def compact(self, messages: list[dict], control: str, archive: ContextArchive) -> list[dict]:
        self.last_info = {}
        out = [m for m in messages if not _is_note(m)]
        starts = _turn_starts(out)
        if len(starts) <= self.keep_turns:
            return self._fallback(list(messages), control, archive)
        cut = starts[-self.keep_turns]
        calls = {
            c["id"]: c for m in out if m["role"] == "assistant" for c in m.get("tool_calls") or []
        }
        budget = self.threshold * self.target
        total = estimate_tokens(out)
        plan, freed = [], 0
        autos = sum(k.startswith("auto-") for k in archive.entries)
        for i in range(1, cut):
            m = out[i]
            call = calls.get(m.get("tool_call_id"), {})
            name = m.get("name") or call.get("name", "?")
            if not self._maskable(m, name, archive):
                continue
            if total - freed <= budget:
                break
            if m["role"] == "user":
                autos += 1
                entry_id = f"auto-{autos}"
                head = _short(str(m["content"]), 160)
                stub = f'{head} [elided to save context; call {RECALL}(id="{entry_id}")]'
            else:
                entry_id = m["tool_call_id"]
                stub = self.stub(entry_id, name, call.get("args"))
            gain = estimate_tokens([m]) - estimate_tokens([{**m, "content": stub}])
            if gain > 0:
                plan.append((i, entry_id, stub))
                freed += gain
        if freed < self.clear_at_least:
            return self._fallback(list(messages), control, archive)
        for i, entry_id, stub in plan:
            archive.put(entry_id, out[i]["content"])
            out[i] = {**out[i], "content": stub}
        self.last_info = {"masked": len(plan)}
        if not self.recite:
            out.append({"role": "user", "content": f"[Acuto] {control}"})
        return self._fallback(out, control, archive)

    def _fallback(self, messages: list[dict], control: str, archive) -> list[dict]:
        if self.fallback is None or estimate_tokens(messages) <= self.threshold:
            return messages
        out = self.fallback.compact(messages, control, archive)
        self.last_info = {**self.last_info, **getattr(self.fallback, "last_info", {})}
        return out


SUMMARY_SYSTEM = (
    "You compress an agent's earlier turns into a working state for the same agent. "
    "Do NOT continue the task or the conversation. ONLY output the summary."
)

SUMMARY_PROMPT = """\
Below are earlier turns of an agent's run. Write the agent's working state so it can
continue from the turns that follow them without re-reading these.

Treat the summary as a working state, not a transcript. For each candidate fact ask:
will a later step read this and act differently? If not, drop it. Once a value is
derived (a count, a total, an id, a classification), carry the result, not its inputs.
Copy exactly every literal the agent will pass to a call: ids, tokens, amounts, names,
paths, parameter names that differ from a naive guess.

Use these sections, and drop a section that would be empty:
1. Position: the last action taken, its outcome, and what the agent was about to do.
2. Values: one line per runtime value still needed: name, exact value, purpose.
3. Done: completed steps, one line each, with their outcome.
4. Next steps: numbered, with the exact call and arguments where known.
5. Lessons: errors met and what fixed them (a wrong parameter name, a failing call).

Do not state which actions are permitted or required, which obligations are pending,
or which rules apply: the harness tracks all of that and shows it to the agent
separately. Do not repeat instructions the user gave; they stay in the conversation.

Earlier turns:
"""


def _render(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        if m["role"] == "assistant":
            calls = "; ".join(
                f"{c.get('name')}({json.dumps(c.get('args') or {}, default=str)})"
                for c in m.get("tool_calls") or []
            )
            text = m.get("content") or ""
            lines.append(f"[assistant] {text}" + (f" -> calls {calls}" if calls else ""))
        elif m["role"] == "tool":
            lines.append(f"[result of {m.get('name', '?')}] {m.get('content')}")
        else:
            lines.append(f"[{m['role']}] {m.get('content')}")
    return "\n".join(lines)


_LITERAL = re.compile(r"\b(?=[\w.-]*\d)[\w.-]{4,}\b")


class SummaryCompactor:
    """The last rung: replace the turns before the last ``keep_turns`` with
    a structured summary written by ``model`` (the agent's own model is
    the better summarizer, arXiv:2610.31778). Kept verbatim: the task,
    the user's own mid-session messages, and the raw tail; a summary
    loses the agent's place sooner than its facts, and a recent turn kept
    raw is followed more faithfully than one folded in (arXiv:2608.06503).

    The summary never carries compliance state: the prompt forbids it, and
    the controller's state is recited after it. Before it is used, every
    id-like literal in it (a token with a digit) must occur in the span or
    in the archive; a summary that invents one is rejected and the span
    is truncated instead (``last_info`` says which). The span itself goes
    to the archive, so the note names a recall id.

    Between ``threshold`` and ``hard_threshold`` (default 1.25 x) the loop
    compacts only at a boundary, when no obligation is pending, so a
    summary is not written in the middle of a unit of work.

    ``keep_user=False`` folds the user's mid-session messages into the
    summary as well, as most agent frameworks do; it is the baseline for
    measuring what keeping them, and keeping rules out of the context,
    is worth."""

    def __init__(
        self,
        model: Model,
        threshold: int = 100_000,
        keep_turns: int = 10,
        *,
        hard_threshold: int | None = None,
        recite: bool = True,
        keep_user: bool = True,
    ) -> None:
        if keep_turns < 1:
            raise ValueError("keep_turns must be at least 1")
        self.model = model
        self.threshold = threshold
        self.keep_turns = keep_turns
        self.hard_threshold = hard_threshold or int(threshold * 1.25)
        self.recite = recite
        self.keep_user = keep_user
        self.last_info: dict = {}

    def compact(self, messages: list[dict], control: str, archive: ContextArchive) -> list[dict]:
        self.last_info = {}
        starts = _turn_starts(messages)
        if len(starts) <= self.keep_turns:
            return list(messages)
        cut = starts[-self.keep_turns]
        keep = _said_by_user if self.keep_user else (lambda m: False)
        span = [m for m in messages[1:cut] if not keep(m) and not _is_note(m)]
        kept = [m for m in messages[1:cut] if keep(m)]
        transcript = _render(span)
        turn = self.model.generate(
            [{"role": "user", "content": SUMMARY_PROMPT + transcript}],
            tools=[],
            system=SUMMARY_SYSTEM,
        )
        summary = (turn.text or "").strip()
        corpus = transcript + "".join(archive.entries.values())
        invented = sorted({t for t in _LITERAL.findall(summary) if t not in corpus})
        span_id = archive.new_span(span)
        if not summary or invented:
            self.last_info = {"summary": "rejected", "invented": invented}
            body = (
                f"[Acuto] {len(span)} earlier messages were removed to fit the context "
                f'window; call {RECALL}(id="{span_id}") to read them.'
            )
        else:
            self.last_info = {"summary": "used"}
            body = (
                f"[Acuto] Summary of {len(span)} earlier messages (reference only: written "
                f"by the model from the turns themselves; Acuto's control state and the "
                f'tool results are authoritative). Full text: {RECALL}(id="{span_id}").\n'
                f"{summary}"
            )
        if not self.recite:
            body += f"\n{control}"
        return messages[:1] + kept + [{"role": "user", "content": body}] + messages[cut:]


@dataclass
class Frame:
    """One step of the run, as the user sees it."""

    step: int
    kind: str  # "neural_choice" | "symbolic_execute" | "idle" | "dead_end" | "final"
    exposed: list[str] = field(default_factory=list)
    masked: list[dict] = field(default_factory=list)  # {tool, contracts}
    action: ProposedAction | None = None
    reason: str = ""
    text: str | None = None
    blocked: list[dict] = field(default_factory=list)  # {tool, reasons}
    records: list[dict] = field(default_factory=list)  # raw tracer records of this step


@dataclass
class RunResult:
    status: str  # "completed" | "dead_end" | "max_steps"
    final_text: str | None
    steps: int
    model_calls: int
    symbolic_steps: int
    neural_action_steps: int
    blocked_proposals: int
    finalize: dict
    controller: SymbolicController
    flow: Any
    tracer: Tracer
    # Condition-independent compliance audit (same contract library run in
    # observe mode over the committed actions of ANY condition). None when
    # no auditor was attached.
    audit: dict | None = None
    # Wall-clock breakdown (seconds): where the time actually went.
    time_model_s: float = 0.0
    time_controller_s: float = 0.0
    time_tools_s: float = 0.0
    # Where the run stopped, to continue it: ``iterate(..., resume=...)``.
    # None when the flow cannot snapshot itself.
    checkpoint: Checkpoint | None = None

    @property
    def symbolic_execution_ratio(self) -> float:
        total = self.symbolic_steps + self.neural_action_steps
        return self.symbolic_steps / total if total else 0.0

    @property
    def neural_decision_ratio(self) -> float:
        total = self.symbolic_steps + self.model_calls
        return self.model_calls / total if total else 0.0


def iterate(
    task: str,
    *,
    model: Model,
    tools: ToolRegistry,
    contracts: Any = None,
    controller: Controller | SymbolicController | None = None,
    flow: Flow | None = None,
    system: str | None = None,
    max_steps: int = 24,
    trace_path: str | None = None,
    facts: dict | None = None,
    predicate_evaluator: Any = None,
    auditor: SymbolicController | None = None,
    control_mode: str = "full",
    probe_cache=None,
    mask_granularity: str = "step",
    compactor: Compactor | None = None,
    resume: Checkpoint | None = None,
    checkpoint_path: str | None = None,
    on_in_flight: str = "raise",
) -> Generator[Frame, None, RunResult]:
    """``resume`` continues a run from its :class:`~acuto.checkpoint.Checkpoint`
    (``RunResult.checkpoint``, or one loaded from ``checkpoint_path``) for
    up to ``max_steps`` more steps: the journal is replayed into the
    controller, which must be fresh, and the flow and context archive are
    restored. ``contracts`` may differ from the original run's; the replay
    then judges the recorded trace under the new ones. Changed tool
    schemas are refused.

    ``checkpoint_path`` makes the run crash-safe at step granularity: the
    checkpoint is written there before every step and records each call
    the step sends before sending it. Resuming a step that was cut off
    after a call to a tool not marked ``idempotent`` raises
    :class:`~acuto.checkpoint.InFlightError` unless
    ``on_in_flight="redo"``."""
    # The loop is written against the Controller facade (acuto/controller.py),
    # the same object a custom loop embeds; ``controller`` below is the
    # underlying SymbolicController, kept for the RunResult.
    if controller is None:
        ctl = Controller(
            contracts,
            tools=tools,
            tracer=Tracer(trace_path),
            predicate_evaluator=predicate_evaluator,
            control_mode=control_mode,
            cache=probe_cache,
            mask_granularity=mask_granularity,
        )
    elif isinstance(controller, Controller):
        ctl = controller
        if ctl.registry is None:
            ctl = Controller.wrap(controller.symbolic, tools=tools)
    else:
        ctl = Controller.wrap(controller, tools=tools)
    controller = ctl.symbolic
    tracer = controller.tracer  # one stream for controller + loop events
    executor = ctl.executor

    flow = flow or FreeFlow()
    archive: ContextArchive | None = None
    if compactor is not None:
        if not hasattr(flow, "messages"):
            raise TypeError("a compactor needs a flow that keeps a `messages` list")
        if RECALL in tools:
            raise ValueError(f"tool name {RECALL!r} is reserved for the compactor")
        archive = ContextArchive(resume.archive if resume is not None else None)
    admit = getattr(compactor, "admit", None)
    recite = getattr(compactor, "recite", False)
    recited: str | None = None  # the control state as last appended
    # Provider-reported context size of the last request, and how many
    # messages that request carried.
    measured, counted = None, 0
    flow.reset(task)

    model_calls = symbolic_steps = neural_action_steps = blocked = 0
    start = 0
    if resume is not None:
        if controller.journal:
            raise ValueError("resume needs a fresh controller")
        if facts:
            raise ValueError("a resumed run takes its facts from the checkpoint")
        if not hasattr(flow, "restore"):
            raise TypeError("resume needs a flow with snapshot()/restore()")
        tracer.record("checkpoint/replay", step=resume.step, entries=len(resume.journal))
        controller.replay(resume.journal)
        check_stamps(resume.stamps, controller, tools, tracer, step=resume.step)
        if resume.in_flight:
            if on_in_flight != "redo" and not all(c["idempotent"] for c in resume.in_flight):
                raise InFlightError(resume.in_flight)
            tracer.record("checkpoint/in_flight", step=resume.step, calls=resume.in_flight)
        if auditor is not None and resume.auditor_journal is not None:
            auditor.replay(resume.auditor_journal)
        flow.restore(resume.flow)
        c = resume.counters
        model_calls = c.get("model_calls", 0)
        symbolic_steps = c.get("symbolic_steps", 0)
        neural_action_steps = c.get("neural_action_steps", 0)
        blocked = c.get("blocked", 0)
        start = resume.step
        recited = c.get("recited")

    for predicate, value in (facts or {}).items():
        controller.assert_fact(predicate, value, source="<initial>")
        if auditor is not None:
            auditor.assert_fact(predicate, value, source="<initial>")

    status, final_text = "max_steps", None
    step = start - 1
    nudges = 0
    max_nudges = 2  # completion-gate retries before giving up as incomplete
    t_model = t_ctrl = t_tools = 0.0

    def _commit(action, result, state):
        """Commit the executed call; verify its return event and decide
        what the model sees (:meth:`Controller.commit`). The call event
        always commits. A return that would falsify an assumption is
        suppressed: the controller writes nothing from it, and the model
        receives a withheld-result error in place of the result (the
        auditor still records the real result)."""
        shown = ctl.commit(action, result, state, step=step)
        if auditor is not None:
            auditor.update(action, result, state)
        return shown

    stamps = {"contracts": contract_stamp(controller), "tools": tool_stamp(tools)}
    top: Checkpoint | None = None  # this step's checkpoint, as written at its start
    sent: list[dict] = []  # calls this step has sent

    def _snapshot(next_step: int) -> Checkpoint:
        return Checkpoint(
            task=task,
            step=next_step,
            journal=list(controller.journal),
            flow=flow.snapshot(),
            archive=dict(archive.entries) if archive is not None else {},
            counters={
                "model_calls": model_calls,
                "symbolic_steps": symbolic_steps,
                "neural_action_steps": neural_action_steps,
                "blocked": blocked,
                "recited": recited,
            },
            auditor_journal=list(auditor.journal) if auditor is not None else None,
            stamps=stamps,
        )

    def _send(action):
        """Record a call before it reaches the environment (write-ahead)."""
        if checkpoint_path is None:
            return
        meta = tools.get(action.tool).metadata if action.tool in tools else {}
        sent.append(
            {
                "tool": action.tool,
                "args": dict(action.args),
                "idempotent": bool(meta.get("idempotent")),
            }
        )
        Checkpoint(**{**top.__dict__, "in_flight": list(sent)}).save(checkpoint_path)

    def _ctx(action=None, result=None, step=0):
        return FlowContext(
            facts=controller.facts,
            state=flow.get_state(),
            step=step,
            last_action=action,
            last_result=result,
        )

    if checkpoint_path is not None and not hasattr(flow, "snapshot"):
        raise TypeError("checkpoint_path needs a flow with snapshot()/restore()")
    for step in range(start, start + max_steps):
        if checkpoint_path is not None:
            top, sent = _snapshot(step), []
            top.save(checkpoint_path)
        if flow.finished():
            status = "completed"
            break

        rec_start = len(tracer.records)
        flow_state = flow.get_state()
        candidates = flow.available_actions(tools.names())
        _t0 = time.perf_counter()
        decision = controller.next_decision(candidates, flow_state)
        t_ctrl += time.perf_counter() - _t0
        tracer.record(
            "controller/decision",
            step=step,
            decision=decision.kind.value,
            actions=decision.actions,
            action=str(decision.action) if decision.action else None,
            reason=decision.reason,
        )
        masked = [
            {"tool": r["tool"], "contracts": r["contracts"]}
            for r in tracer.records[rec_start:]
            if r["kind"] == "action/masked"
        ]

        if decision.kind is StepKind.DEAD_END and getattr(model, "idles", False):
            # A model that can wait still has a move
            # when no tool is admissible: an idle step.
            decision = StepDecision.neural([], reason=decision.reason)

        if decision.kind is StepKind.DEAD_END:
            tracer.record("controller/dead_end", step=step, reason=decision.reason)
            status = "dead_end"
            yield Frame(
                step=step,
                kind="dead_end",
                masked=masked,
                reason=decision.reason,
                records=tracer.records[rec_start:],
            )
            break

        if decision.kind is StepKind.SYMBOLIC_EXECUTE:
            action = decision.action
            _t0 = time.perf_counter()
            verdict = controller.validate(action, flow_state)  # hard boundary, always
            t_ctrl += time.perf_counter() - _t0
            if not verdict.allowed:
                # The fast-forward prediction is withheld: it never reaches the
                # environment, and the model decides this step over the
                # predicted tool (its own call is verified as usual).
                tracer.record(
                    "action/blocked", step=step, tool=action.tool, reasons=verdict.reasons
                )
                tracer.record(
                    "controller/fallback",
                    step=step,
                    tool=action.tool,
                    reason=f"fast-forward prediction blocked: {verdict.reasons}",
                )
                decision = StepDecision.neural(
                    [action.tool], reason=f"fast-forward prediction blocked: {verdict.reasons}"
                )
            else:
                _t0 = time.perf_counter()
                _send(action)
                result = executor.execute(action)
                t_tools += time.perf_counter() - _t0
                symbolic_steps += 1
                tracer.record(
                    "action/symbolic", step=step, tool=action.tool, args=action.args, ok=result.ok
                )
                shown = _commit(action, result, flow_state)
                flow.notify_symbolic(action, shown, decision.reason)
                flow.advance(_ctx(action, shown, step))
                yield Frame(
                    step=step,
                    kind="symbolic_execute",
                    masked=masked,
                    action=action,
                    reason=decision.reason,
                    records=tracer.records[rec_start:],
                )
                continue

        # NEURAL_CHOICE — dynamic per-step tool exposure.
        if getattr(controller, "mask_granularity", "step") == "hint":
            # Cache-friendly soft masking: the tools block stays stable;
            # the restriction rides in an appended message (validate() is
            # still the hard boundary).
            allowed = getattr(controller, "last_admissible", None)
            if allowed is not None and set(allowed) != set(decision.actions):
                flow.nudge("[Acuto] Currently permitted tools: " + ", ".join(allowed) + ".")
        messages = flow.build_context()
        exposed = tools.schemas(decision.actions)
        if archive is not None:
            used = context_tokens(messages, measured, counted)
            hard = getattr(compactor, "hard_threshold", compactor.threshold)
            # Between the thresholds, compact only at a boundary: no
            # obligation open, so no unit of work is cut in the middle.
            if used > hard or (used > compactor.threshold and not controller.obligations.pending()):
                compacted = compactor.compact(
                    messages, control_state_text(controller, list(decision.actions)), archive
                )
                if compacted != messages:
                    flow.messages = compacted
                    messages = flow.build_context()
                    measured, recited = None, None
                    tracer.record(
                        "context/compacted",
                        step=step,
                        before=used,
                        after=estimate_tokens(messages),
                        archived=len(archive),
                        **getattr(compactor, "last_info", {}),
                    )
            if recite:
                # Append-only: a fresh copy whenever the state changes, the
                # old copies left in place (deleting one would rewrite the
                # cached prefix); compaction drops the superseded ones.
                state_text = control_state_text(controller)
                if state_text != recited:
                    flow.nudge(f"[Acuto] {state_text}")
                    messages = flow.build_context()
                    recited = state_text
            # Declared from step 0: a tools block that grows mid-run would
            # invalidate the provider's cached prompt prefix.
            exposed = exposed + [archive.schema()]
        _t0 = time.perf_counter()
        turn = model.generate(messages, tools=exposed, system=system)
        t_model += time.perf_counter() - _t0
        model_calls += 1
        reported = (turn.usage or {}).get("context") or (turn.usage or {}).get("prompt")
        if reported:
            measured, counted = reported, len(messages)
        tracer.record(
            "model/response",
            step=step,
            text=turn.text,
            tool_calls=[{"name": c.name, "args": c.args} for c in turn.tool_calls],
            usage=turn.usage,
        )
        if turn.idle and not turn.tool_calls:
            violated = controller.idle(flow_state)
            tracer.record("action/idle", step=step, violated=[c.name for c in violated])
            yield Frame(
                step=step,
                kind="idle",
                exposed=decision.actions,
                masked=masked,
                records=tracer.records[rec_start:],
            )
            continue
        flow.update_model_turn(turn)

        if not turn.tool_calls:
            # Floor-yield gate: the episode can only end after a text turn,
            # so no text turn is allowed out while the obligation ledger is
            # non-empty, a *next* obligation (one the model was handed a
            # singleton action space for) as much as an eventually one.
            # Jump if the args are determined; otherwise nudge with the
            # obligation named.
            pending_ev = controller.obligations.due() + controller.obligations.due_eventually()
            if pending_ev and nudges < max_nudges:
                ob = pending_ev[0]
                action = ctl.due_obligation(flow.get_state())
                if action is not None:
                    verdict = controller.validate(action, flow_state)
                    if verdict.allowed:
                        _t0 = time.perf_counter()
                        _send(action)
                        result = executor.execute(action)
                        t_tools += time.perf_counter() - _t0
                        symbolic_steps += 1
                        tracer.record(
                            "action/symbolic",
                            step=step,
                            tool=action.tool,
                            args=action.args,
                            ok=result.ok,
                            reason="floor-yield obligation",
                        )
                        shown = _commit(action, result, flow_state)
                        flow.notify_symbolic(
                            action, shown, f"pending obligation before finishing: {ob.desc}"
                        )
                        flow.advance(_ctx(action, shown, step))
                        yield Frame(
                            step=step,
                            kind="symbolic_execute",
                            masked=masked,
                            action=action,
                            reason=f"floor-yield obligation: {ob.desc}",
                            records=tracer.records[rec_start:],
                        )
                        continue
                if nudges >= max_nudges:
                    # Bounded like the completion gate: a model that answers
                    # the obligation nudge with text (or nothing) forever must
                    # not be re-prompted until max_steps — end with the
                    # obligation reported as pending.
                    status, final_text = "completed", turn.text
                    yield Frame(
                        step=step,
                        kind="final",
                        exposed=decision.actions,
                        masked=masked,
                        text=turn.text,
                        records=tracer.records[rec_start:],
                    )
                    break
                nudges += 1
                tracer.record(
                    "controller/nudge", step=step, reason=f"pending obligation: {ob.desc}"
                )
                flow.nudge(
                    f"[Acuto] Before finishing, you must still: {ob.desc}. Complete it now using the tools."
                )
                yield Frame(
                    step=step,
                    kind="neural_choice",
                    exposed=decision.actions,
                    masked=masked,
                    text=turn.text,
                    records=tracer.records[rec_start:],
                )
                continue
            owed = controller.owed_rules() if nudges < max_nudges else []
            names = [controller.rules[r]["name"] for r in owed]
            if nudges < max_nudges and hasattr(controller, "owed_deliverables"):
                names += controller.owed_deliverables(flow_state)
            if names:
                # A confirmed rule or a deliverable still demands a call (an
                # eventuality the session would end with open): same
                # bounded gate.
                nudges += 1
                tracer.record("controller/nudge", step=step, reason=f"owed by {names}")
                flow.nudge(
                    "[Acuto] Before finishing, these rules still require a call: "
                    + "; ".join(names)
                    + ". Complete them now using the tools."
                )
                yield Frame(
                    step=step,
                    kind="neural_choice",
                    exposed=decision.actions,
                    masked=masked,
                    text=turn.text,
                    records=tracer.records[rec_start:],
                )
                continue
            if (
                getattr(flow, "requires_terminal", False)
                and not flow.finished()
                and nudges < max_nudges
            ):
                # Procedural completion gate: a text-only answer does not
                # end a flow that declares terminal states — the model is
                # nudged back to finish the procedure via tools. Bounded:
                # after max_nudges the run ends (status reflects the truth).
                nudges += 1
                tracer.record(
                    "controller/nudge", step=step, reason="text answer before terminal state"
                )
                flow.nudge(
                    "[Acuto] The task is not complete: finish the procedure using "
                    "the available tools (including submitting the final result)."
                )
                yield Frame(
                    step=step,
                    kind="neural_choice",
                    exposed=decision.actions,
                    masked=masked,
                    text=turn.text,
                    records=tracer.records[rec_start:],
                )
                continue
            status, final_text = "completed", turn.text
            yield Frame(
                step=step,
                kind="final",
                exposed=decision.actions,
                masked=masked,
                text=turn.text,
                records=tracer.records[rec_start:],
            )
            break

        frame_blocked: list[dict] = []
        last_action = last_result = None
        for call in turn.tool_calls:
            if archive is not None and call.name == RECALL:
                # A view operation on the history, not an action: the
                # controller never sees it and no contract steps on it.
                cid = str((call.args or {}).get("id", ""))
                limit = 4 * (getattr(compactor, "max_result_tokens", None) or 20_000)
                content = archive.page(cid, int((call.args or {}).get("offset") or 0), limit)
                tracer.record("context/recalled", step=step, id=cid, ok=content is not None)
                flow.add_tool_result(
                    call,
                    content
                    if content is not None
                    else json.dumps({"ok": False, "error": f"no elided result with id {cid!r}"}),
                )
                continue
            if call.name not in tools:
                # A hallucinated tool name is a real proposal that failed —
                # trace it so loops of invented names are visible, not silent.
                tracer.record("action/unknown", step=step, tool=call.name)
                frame_blocked.append({"tool": call.name, "reasons": ["unknown tool"]})
                # Actionable feedback: a bare "unknown tool" invites another
                # guess (observed: 10+ invented names per row); naming the
                # currently permitted tools closes the loop. The text also
                # says the name must come from that list: with the list
                # alone Sonnet kept permuting names of exposed tools
                # (aircraft, traced 2026-09).
                flow.add_tool_result(
                    call,
                    json.dumps(
                        {
                            "ok": False,
                            "error": (
                                f"unknown tool {call.name}: choose one of available_tools, "
                                "spelled exactly as listed; no other tool exists"
                            ),
                            "available_tools": list(decision.actions),
                        }
                    ),
                )
                continue
            call_state = flow.get_state()
            proposed = ToolCall(name=call.name, args=call.args, id=call.id)
            action = ProposedAction(call.name, dict(call.args or {}))
            tracer.record("action/proposed", step=step, tool=action.tool, args=action.args)
            # Prescriptive stage boundary: when the flow scopes the current
            # state to a tool subset, an out-of-stage proposal is invalid by
            # the workflow medium's own semantics — enforced here so it does
            # not depend on a redundant external contract existing.
            stage_tools = flow.available_actions(tools.names())
            if call.name not in stage_tools:
                blocked += 1
                frame_blocked.append(
                    {"tool": action.tool, "reasons": ["outside current workflow stage"]}
                )
                tracer.record(
                    "action/blocked",
                    step=step,
                    tool=action.tool,
                    verdict="block",
                    reasons=["workflow-stage boundary"],
                )
                flow.add_tool_result(
                    proposed,
                    json.dumps(
                        {
                            "ok": False,
                            "error": "This action is not available in the current workflow stage.",
                        }
                    ),
                )
                continue
            _t0 = time.perf_counter()
            verdict = controller.validate(action, call_state)
            t_ctrl += time.perf_counter() - _t0
            if verdict.kind is DecisionKind.ALLOW:
                _t0 = time.perf_counter()
                _send(action)
                result = executor.execute(action)
                t_tools += time.perf_counter() - _t0
                neural_action_steps += 1
                tracer.record("tool/result", step=step, tool=action.tool, ok=result.ok)
                shown = _commit(action, result, call_state)
                content = json.dumps(shown.payload(), default=str)
                if verdict.warnings:
                    content += (
                        "\n[Acuto] This call goes against "
                        + "; ".join(verdict.warnings)
                        + ". The user has not confirmed that rule, so the call ran; "
                        "check with the user if it matters."
                    )
                if admit is not None:
                    content = admit(call.id, call.name, call.args, content, archive)
                flow.add_tool_result(proposed, content)
                last_action, last_result = action, shown
            else:
                blocked += 1
                frame_blocked.append({"tool": action.tool, "reasons": verdict.reasons})
                tracer.record(
                    "action/blocked",
                    step=step,
                    tool=action.tool,
                    args=dict(action.args or {}),
                    verdict=verdict.kind.value,
                    reasons=verdict.reasons,
                )
                flow.receive_control_feedback(proposed, verdict)
        if last_action is not None:
            flow.advance(_ctx(last_action, last_result, step))
        yield Frame(
            step=step,
            kind="neural_choice",
            exposed=decision.actions,
            masked=masked,
            blocked=frame_blocked,
            text=turn.text,
            records=tracer.records[rec_start:],
        )

    finalize = controller.finalize()
    audit = None
    if auditor is not None:
        # LTLf safety monitors latch (once violated, every later event
        # re-reports) — the audit counts each rule's FIRST firing only.
        seen: set = set()
        fired = []
        for r in auditor.tracer.by_kind("contract/violation"):
            new = [x for x in r.get("contracts", []) if x not in seen]
            if new:
                seen.update(new)
                fired.append({"tool": r.get("tool"), "contracts": new})
        audit_final = auditor.finalize()
        audit = {
            "committed_violations": fired,
            "violation_count": sum(len(f["contracts"]) for f in fired),
            "proc_clean": not fired
            and not audit_final["ltlf_violations"]
            and not audit_final["pending_obligations"],
            "finalize": audit_final,
        }
    checkpoint = _snapshot(step + 1) if hasattr(flow, "snapshot") else None
    if checkpoint_path is not None:
        checkpoint.save(checkpoint_path)
    return RunResult(
        status=status,
        final_text=final_text,
        steps=step + 1,
        model_calls=model_calls,
        symbolic_steps=symbolic_steps,
        neural_action_steps=neural_action_steps,
        blocked_proposals=blocked,
        finalize=finalize,
        controller=controller,
        flow=flow,
        tracer=tracer,
        audit=audit,
        time_model_s=t_model,
        time_controller_s=t_ctrl,
        time_tools_s=t_tools,
        checkpoint=checkpoint,
    )


def run(task: str, *, verbose: bool = False, **kwargs: Any) -> RunResult:
    """Drain :func:`iterate` and return the RunResult. ``verbose=True``
    prints every step and a summary (:mod:`acuto.console`)."""
    console = None
    if verbose:
        from acuto.console import Console

        console = Console()
    gen = iterate(task, **kwargs)
    while True:
        try:
            frame = next(gen)
        except StopIteration as stop:
            if console is not None:
                console.summary(stop.value)
            return stop.value
        if console is not None:
            console.step(frame)
