"""Obligations: what MUST happen, as opposed to what MAY happen.

An obligation is active procedural intent owned by the controller. It is
not enforced through the LTLf layer alone: a monitor decides an
undischarged X/F obligation only when the session ends (ContrAgent reads
what is still owed then on idle events after the last one, so it fails),
which is the right semantics for a checker but comes too late to get the
call made. Acuto therefore tracks obligations as first-class metadata:
the controller *executes* them (jump-forward) when their arguments are
procedurally determined, hands the LLM a singleton action space when they
are not, keeps a text turn from ending the session while any is pending,
and reports any still-pending obligation at session end as a violation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from acuto.facts import PredicateContext


@dataclass(frozen=True)
class ObligationSpec:
    """Compiled form of ``when(fact).obligates(tool, ...)``.

    ``args`` is a constant argument dict; ``binder`` is a *deterministic*
    function ``PredicateContext -> dict | None`` that binds arguments
    from facts / agent state. A binder must not do semantic reasoning —
    if it cannot determine the arguments it returns None and control
    stays with the LLM (restricted to this one tool).
    """

    trigger: str  # fact predicate that activates the obligation
    tool: str
    args: dict[str, Any] | None = None
    binder: Callable[[PredicateContext], dict | None] | None = None
    deadline: str = "next"  # "next" = preempt now; "eventually" = floor-yield gate
    desc: str = ""
    # A call of ``tool`` that ran and returned an error discharges it too, as
    # it does an obligation read off ``called(tool)`` in the contracts.
    discharged_on_error: bool = False

    @property
    def name(self) -> str:
        return self.desc or f"when {self.trigger}: must call {self.tool}"


@dataclass
class ActiveObligation:
    spec: ObligationSpec
    triggered_at: int  # controller step index when the trigger fired
    discharged: bool = False

    @property
    def tool(self) -> str:
        return self.spec.tool

    @property
    def desc(self) -> str:
        return self.spec.name


class ObligationEngine:
    """Tracks activation and discharge of obligations."""

    def __init__(self, specs: list[ObligationSpec] | None = None) -> None:
        self._specs = list(specs or [])
        self._active: list[ActiveObligation] = []
        self._history: list[ActiveObligation] = []

    def specs(self) -> list[ObligationSpec]:
        return list(self._specs)

    def on_fact_asserted(self, predicate: str, *, step: int) -> list[ActiveObligation]:
        """Activate obligations whose trigger fact just became true."""
        activated = []
        for spec in self._specs:
            if spec.trigger != predicate:
                continue
            if any(o.spec is spec for o in self._active):
                continue  # already pending
            ob = ActiveObligation(spec=spec, triggered_at=step)
            self._active.append(ob)
            activated.append(ob)
        return activated

    def on_action_executed(self, tool: str, ok: bool) -> list[ActiveObligation]:
        """Discharge pending obligations satisfied by a call: a successful
        one, or any that ran for an obligation discharged on error."""
        discharged = []
        remaining = []
        for ob in self._active:
            if ob.spec.tool == tool and (ok or ob.spec.discharged_on_error):
                ob.discharged = True
                self._history.append(ob)
                discharged.append(ob)
            else:
                remaining.append(ob)
        self._active = remaining
        return discharged

    def due(self) -> list[ActiveObligation]:
        """Preempting obligations (deadline="next"): symbolic execution
        takes over immediately."""
        return [o for o in self._active if o.spec.deadline == "next"]

    def due_eventually(self) -> list[ActiveObligation]:
        """Deferred obligations (deadline="eventually", e.g. F-shaped
        response contracts): they do not preempt, but the floor-yield gate
        refuses to hand back a text turn while any remain (or any next
        obligation does)."""
        return [o for o in self._active if o.spec.deadline != "next"]

    def pending(self) -> list[ActiveObligation]:
        return list(self._active)

    @staticmethod
    def resolve_args(ob: ActiveObligation, context: PredicateContext) -> dict | None:
        """Deterministically bind the obligation's arguments, or None."""
        if ob.spec.args is not None:
            return dict(ob.spec.args)
        if ob.spec.binder is not None:
            return ob.spec.binder(context)
        return None
