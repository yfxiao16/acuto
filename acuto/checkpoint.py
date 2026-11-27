"""Checkpoint and resume of one agent run (docs/TODO.md A1).

A checkpoint is the run's log, not a dump of its state: the controller's
journal (every input that moved the symbolic control state), the flow's
messages, the context archive (what compaction took out of view), the
step counter and the loop's counters. :func:`acuto.loop.iterate` with
``resume=`` replays the journal into a fresh controller and continues
from the next step, so residuals, facts and obligations are recomputed
rather than deserialized. The same replay under a changed contract
library is how a changed role contract is swapped in: the new contracts
judge the trace that already happened.

A step is the unit of atomicity. With ``checkpoint_path`` the loop
writes the checkpoint before every step, and before every tool call it
records the calls the current step has sent (``in_flight``). A crash
mid-step leaves the checkpoint of that step's start plus those calls:
redoing the step is safe when every one of them is idempotent
(``Tool.metadata["idempotent"]``); otherwise resuming raises
:class:`InFlightError` unless the caller accepts the redo.

``stamps`` fingerprint the contract library and the tool schemas. On
resume, changed contracts are allowed (the replay judges the trace under
them, and the change is traced); changed tool schemas are refused, since
recorded calls may no longer mean what they meant.

``save``/``load`` write JSON, atomically. Tool outputs and flow states
that are not JSON values are written with ``str``; keep them
JSON-serializable when a checkpoint must survive the process.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

VERSION = 1


class InFlightError(RuntimeError):
    """The checkpoint was written mid-step: calls in ``calls`` were sent
    to the environment and may have taken effect, and at least one of
    them is not idempotent, so redoing the step could repeat it."""

    def __init__(self, calls: list[dict], where: str = "step") -> None:
        self.calls = calls
        unsafe = [c["tool"] for c in calls if not c.get("idempotent")]
        super().__init__(
            f"the checkpointed {where} was interrupted after sending {unsafe} "
            "(not idempotent); pass on_in_flight='redo' to run it again anyway"
        )


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def contract_stamp(controller: Any) -> str:
    """Fingerprint of a controller's contract library: contracts with
    their formulas, obligation specs, and fact extractors."""
    cc = controller.contracts
    # The authored library only: rules attached mid-run are journal entries,
    # replayed with the rest, not part of what the run was started with.
    contracts = [
        (c.name, repr(c.formula), repr(c.assumption))
        for c in controller.backend.contracts()
        if c.rule is None
    ]
    obligations = [(o.trigger, o.tool, o.deadline, o.desc) for o in controller.obligations.specs()]
    extractors = {
        tool: [
            (getattr(x.predicate, "__qualname__", x.predicate), getattr(x, "on_error", False))
            for x in specs
        ]
        for tool, specs in cc.extractors.items()
    }
    return _digest([contracts, obligations, extractors, cc.invalidations])


def tool_stamp(registry: Any) -> str:
    """Fingerprint of the tool schemas a run exposes."""
    return _digest(sorted(registry.schemas(), key=lambda t: t["name"]))


def check_stamps(saved: dict, controller: Any, registry: Any, tracer: Any, **where: Any) -> None:
    """Refuse changed tool schemas; trace changed contracts."""
    if not saved:
        return
    if saved.get("tools") not in (None, tool_stamp(registry)):
        raise ValueError(
            "the tool schemas changed since the checkpoint was written; recorded "
            "calls may no longer mean what they meant"
        )
    if saved.get("contracts") not in (None, contract_stamp(controller)):
        tracer.record("checkpoint/contracts_changed", **where)


def _save_atomic(data: dict, path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, default=str)
    os.replace(tmp, path)


@dataclass
class Checkpoint:
    task: str
    step: int  # the next step to run
    journal: list[dict]
    flow: dict  # Flow.snapshot()
    archive: dict[str, str] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    auditor_journal: list[dict] | None = None
    in_flight: list[dict] = field(default_factory=list)  # calls sent this step
    stamps: dict[str, str] = field(default_factory=dict)
    version: int = VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        if data.get("version", VERSION) != VERSION:
            raise ValueError(f"checkpoint version {data.get('version')} is not {VERSION}")
        return cls(**data)

    def save(self, path: str) -> None:
        _save_atomic(self.to_dict(), path)

    @classmethod
    def load(cls, path: str) -> Checkpoint:
        with open(path) as f:
            return cls.from_dict(json.load(f))
