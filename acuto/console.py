"""Terminal rendering of a run: what Acuto decided at every step.

``agent.run(task, verbose=True)`` prints a banner, one block per step
as it happens, and a summary. The same renderer replays a finished
:class:`~acuto.loop.RunResult` (:func:`render`), so a trace can be read
back after the fact. Colour is used when the stream is a terminal and
``NO_COLOR`` is unset; pass ``color=False`` to force plain text.

A step block names who acted (``FAST-FORWARD`` for a call the controller
executed with no model call, ``MODEL`` for a model turn, ``DEAD END``),
then what followed: what was masked, what the model proposed, whether
each proposal ran or was refused (``REQUIRE`` names the missing
prerequisite, ``BLOCK`` is final, ``WITHHELD`` marks a result kept from
the model), and the facts and obligations the step produced.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, TextIO

__all__ = ["Console", "render"]

_TAG_WIDTH = 13
_INDENT = " " * 8


def _use_color(stream: TextIO, color: bool | None) -> bool:
    if color is not None:
        return color
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


class _Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

    def bold(self, t: str) -> str:
        return self._wrap("1", t)

    def dim(self, t: str) -> str:
        return self._wrap("2", t)

    def green(self, t: str) -> str:
        return self._wrap("32", t)

    def blue(self, t: str) -> str:
        return self._wrap("34", t)

    def yellow(self, t: str) -> str:
        return self._wrap("33", t)

    def red(self, t: str) -> str:
        return self._wrap("31", t)

    def magenta(self, t: str) -> str:
        return self._wrap("35", t)

    def cyan(self, t: str) -> str:
        return self._wrap("36", t)


def _call(tool: str, args: dict | None) -> str:
    if not args:
        return f"{tool}()"
    inner = ", ".join(f"{k}={json.dumps(v, default=str)}" for k, v in args.items())
    return f"{tool}({inner})"


def _reasons(tool: str, names: list[str]) -> str:
    """Contract names without the tool's own prefix: ``issue_refund requires
    identity_verified`` reads as ``requires identity_verified`` next to the
    tool, and the same reason then groups across tools."""
    out = []
    for n in names:
        n = str(n)
        if n.startswith(tool + " "):
            n = n[len(tool) + 1 :]
        out.append(n)
    return "; ".join(out)


def _shorten(text: str, limit: int = 96) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Console:
    """Renders steps to ``stream`` (default: stdout)."""

    def __init__(
        self, stream: TextIO | None = None, *, color: bool | None = None, width: int = 72
    ) -> None:
        self.stream = stream or sys.stdout
        self.s = _Style(_use_color(self.stream, color))
        self.width = width

    # ------------------------------------------------------------------
    def _w(self, line: str = "") -> None:
        self.stream.write(line + "\n")

    def _tag(self, text: str, paint) -> str:
        return paint(f"{text:<{_TAG_WIDTH}}")

    def _row(self, connector: str, label: str, text: str, paint=None) -> None:
        label = f"{label:<{_TAG_WIDTH}}"
        if paint is not None:
            label = paint(label)
        self._w(f"{_INDENT}{self.s.dim(connector)} {label} {text}")

    # ------------------------------------------------------------------
    def banner(
        self,
        *,
        library: Any = None,
        tools: int | None = None,
        flow: str | None = None,
        model: str | None = None,
        control_mode: str | None = None,
        version: str | None = None,
    ) -> None:
        from acuto import __version__

        s = self.s
        title = f" Acuto {version or __version__} "
        inner = self.width - 2
        top = "╭─" + title + "─" * max(0, inner - len(title) - 1) + "╮"
        lines: list[str] = []
        if library is not None:
            n = len(getattr(library, "contracts", []) or [])
            name = getattr(library, "name", str(library))
            verified = getattr(library, "verified", None)
            status = (
                ""
                if verified is None
                else ("verified" if getattr(verified, "ok", False) else "CONFLICT")
            )
            parts = [name, f"{n} contract{'s' if n != 1 else ''}"]
            if status:
                parts.append(status)
            lines.append(("library", " · ".join(parts)))
        run_parts = []
        if tools is not None:
            run_parts.append(f"{tools} tools")
        if flow:
            run_parts.append(f"flow {flow}")
        if model:
            run_parts.append(f"model {model}")
        if control_mode:
            run_parts.append(f"control {control_mode}")
        if run_parts:
            lines.append(("run", " · ".join(run_parts)))
        self._w(s.dim(top))
        for label, text in lines:
            body = f"  {label:<8} {text}"
            pad = max(0, inner - len(body))
            self._w(s.dim("│") + body + " " * pad + s.dim("│"))
        self._w(s.dim("╰" + "─" * inner + "╯"))
        self._w()

    # ------------------------------------------------------------------
    def step(self, frame: Any) -> None:
        """Render one :class:`~acuto.loop.Frame` from its raw records."""
        self._records(frame.step, list(frame.records))

    def _records(self, step: int, records: list[dict]) -> None:
        s = self.s
        masked = [r for r in records if r["kind"] == "action/masked"]
        decision = next((r for r in records if r["kind"] == "controller/decision"), None)
        head = " " + s.bold(f"{'step ' + str(step):<8}") + " "
        rows: list[tuple[str, str, Any]] = []  # (label, text, paint)

        if decision is None:
            kind = "unknown"
        else:
            kind = decision["decision"]
        if kind == "symbolic_execute" and decision is not None:
            sym = next((r for r in records if r["kind"] == "action/symbolic"), None)
            call = _call(sym["tool"], sym.get("args")) if sym else decision.get("action", "")
            self._w(head + self._tag("FAST-FORWARD", lambda t: s.bold(s.green(t))) + " " + call)
            reason = (sym or {}).get("reason") or decision.get("reason", "")
            if reason:
                rows.append(("reason", s.dim(_shorten(reason)), None))
        elif kind == "neural_choice" and decision is not None:
            n = len(decision.get("actions", []))
            text = f"{n} tool{'s' if n != 1 else ''} exposed"
            self._w(head + self._tag("MODEL", s.blue) + " " + text)
        elif kind == "dead_end" and decision is not None:
            self._w(
                head
                + self._tag("DEAD END", lambda t: s.bold(s.red(t)))
                + " "
                + decision.get("reason", "")
            )
        else:
            self._w(head)

        if masked:
            rows.append(("masked", self._masked_text(masked), None))

        for r in records:
            k = r["kind"]
            if k == "model/response":
                for c in r.get("tool_calls") or []:
                    rows.append(("proposes", _call(c["name"], c.get("args")), None))
                if r.get("text") and not (r.get("tool_calls") or []):
                    rows.append(("says", s.dim(f'"{_shorten(r["text"])}"'), None))
            elif k == "action/unknown":
                rows.append(("UNKNOWN", f"{r['tool']} is not a tool", s.yellow))
            elif k == "action/idle":
                rows.append(("idles", s.dim("no call this step"), None))
            elif k == "context/compacted":
                text = f"context {r['before']} → {r['after']} tokens ({r['archived']} archived)"
                rows.append(("compacted", s.dim(text), None))
            elif k == "rule/attached":
                rows.append(("+ rule", _shorten(r["name"]) + " (proposed)", s.cyan))
            elif k == "rule/confirmed":
                rows.append(("✓ rule", r["rule"] + " confirmed", s.green))
            elif k == "rule/revoked":
                rows.append(("- rule", r["rule"] + " revoked", s.dim))
            elif k == "rule/advisory_violated":
                rows.append(("WARN", "; ".join(r["contracts"]), s.yellow))
            elif k == "context/recalled":
                rows.append(("recalls", r["id"] + ("" if r["ok"] else " (not found)"), None))
            elif k == "action/blocked":
                verdict = r.get("verdict")
                text = f"{r['tool']}: " + _reasons(r["tool"], r.get("reasons", []))
                rows.append(
                    ("REQUIRE", text, s.yellow) if verdict == "require" else ("BLOCK", text, s.red)
                )
            elif k == "tool/result":
                mark = s.green("ok") if r.get("ok") else s.red("error")
                rows.append(("ran", f"{r['tool']} → {mark}", None))
            elif k == "action/symbolic":
                mark = s.green("ok") if r.get("ok") else s.red("error")
                rows.append(("ran", f"{r['tool']} → {mark}", None))
            elif k == "return/withheld":
                rows.append(
                    (
                        "WITHHELD",
                        f"{r['tool']}: result breaks " + "; ".join(r.get("contracts", [])),
                        s.magenta,
                    )
                )
            elif k == "fact/asserted":
                rows.append(
                    (
                        "+ fact",
                        f"{r['predicate']} = {json.dumps(r.get('value'), default=str)}",
                        s.green,
                    )
                )
            elif k == "fact/invalidated":
                rows.append(("- fact", f"{r['predicate']} retracted", s.dim))
            elif k == "fact/suppressed":
                rows.append(
                    (
                        "SUPPRESSED",
                        f"{r['predicate']}: breaks " + "; ".join(r.get("contracts", [])),
                        s.magenta,
                    )
                )
            elif k == "obligation/created":
                rows.append(("! obligation", _shorten(r["obligation"]), s.yellow))
            elif k == "obligation/satisfied":
                rows.append(("✓ obligation", _shorten(r["obligation"]), s.green))
            elif k == "controller/nudge":
                rows.append(("nudge", s.dim(_shorten(r.get("reason", ""))), None))
            elif k == "controller/dead_end" and kind != "dead_end":
                rows.append(("DEAD END", r.get("reason", ""), s.red))

        for i, (label, text, paint) in enumerate(rows):
            self._row("└" if i == len(rows) - 1 else "├", label, text, paint)
        self._w()

    def _masked_text(self, masked: list[dict]) -> str:
        by_reason: dict[str, list[str]] = {}
        for r in masked:
            by_reason.setdefault(_reasons(r["tool"], r.get("contracts", [])), []).append(r["tool"])
        parts = []
        for reason, tools in by_reason.items():
            names = ", ".join(tools[:3]) + (f" +{len(tools) - 3}" if len(tools) > 3 else "")
            parts.append(f"{names} {self.s.dim('(' + _shorten(reason, 60) + ')')}")
        return "  ".join(parts)

    # ------------------------------------------------------------------
    def summary(self, result: Any) -> None:
        s = self.s
        fin = result.finalize or {}
        viols = len(fin.get("ltlf_violations", []))
        pend = len(fin.get("pending_obligations", []))
        status = {"completed": s.green, "dead_end": s.red, "max_steps": s.yellow}.get(
            result.status, str
        )(result.status)
        ff = result.symbolic_steps
        self._w(s.dim("─" * self.width))
        self._w(
            f" {s.bold(status)} · {result.steps} steps · {result.model_calls} model calls · "
            f"{ff} fast-forward{'s' if ff != 1 else ''} "
            f"({result.symbolic_execution_ratio:.0%} of actions)"
        )
        self._w(
            f" {result.blocked_proposals} blocked before execution · "
            f"{viols} violation{'s' if viols != 1 else ''} · "
            f"{pend} obligation{'s' if pend != 1 else ''} pending"
        )
        if result.final_text:
            self._w(f' final  "{_shorten(result.final_text, 200)}"')
        self._w(
            s.dim(
                f" time  model {result.time_model_s:.2f}s · controller "
                f"{result.time_controller_s:.2f}s · tools {result.time_tools_s:.2f}s"
            )
        )
        self._w()

    # ------------------------------------------------------------------
    def render(self, result: Any, *, banner: bool = True) -> None:
        """Replay a finished run from its tracer records."""
        if banner:
            self.banner(
                library=getattr(result, "library", None),
                control_mode=getattr(result.controller, "control_mode", None),
            )
        groups: list[tuple[int, list[dict]]] = []
        pending: list[dict] = []
        for rec in result.tracer.records:
            if rec["kind"] == "controller/decision":
                groups.append((rec["step"], pending + [rec]))
                pending = []
            elif groups and "step" in rec and rec["step"] == groups[-1][0]:
                groups[-1][1].append(rec)
            elif groups and "step" not in rec and rec["kind"] != "action/masked":
                groups[-1][1].append(rec)
            else:
                pending.append(rec)
        for step, records in groups:
            self._records(step, records)
        self.summary(result)


def render(result: Any, *, stream: TextIO | None = None, color: bool | None = None) -> None:
    """Print a finished run the way ``verbose=True`` would have."""
    Console(stream, color=color).render(result)
