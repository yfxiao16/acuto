"""Structured decision types.

The controller never answers with a bare bool. ``Decision`` is the
validation verdict for one concrete action; ``StepDecision`` says who
controls the next step (neural choice / symbolic execution /
dead end). ``REQUIRE`` exists so that missing prerequisites are not
awkwardly encoded as hard blocks — later it can also carry pending
verification, human approval, or semantic clarification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from acuto.obligations import ActiveObligation


@dataclass
class ProposedAction:
    """A concrete (tool, args) pair, before or after validation."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.tool}({self.args})"


class DecisionKind(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE = "require"


@dataclass
class Decision:
    """Verdict of ``controller.validate(action)``.

    * ALLOW   — execute now.
    * REQUIRE — recoverable: prerequisites are missing but can still be
      satisfied in-session. ``requirements`` names the missing facts /
      prerequisite tools; ``hints`` names concrete tools that establish
      them (the deterministic "FIRST call X" recovery channel).
    * BLOCK   — hard: the action violates a rule that retrying or
      rephrasing cannot fix.
    """

    kind: DecisionKind
    reasons: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    # Unconfirmed rules the call goes against: it is still allowed, and the
    # model is told alongside the result.
    warnings: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.kind is DecisionKind.ALLOW

    def message(self, action: ProposedAction | None = None) -> str:
        """Feedback string fed back to the model when not allowed."""
        name = f"`{action.tool}`" if action else "the action"
        if self.kind is DecisionKind.ALLOW:
            return f"{name} is allowed."
        if self.kind is DecisionKind.REQUIRE:
            parts = [f"{name} was blocked because required prerequisites are not satisfied: "]
            parts.append("; ".join(self.reasons))
            if self.hints:
                parts.append(" FIRST " + "; then ".join(self.hints) + ",")
                parts.append(f" THEN retry {name} with the same arguments, in this same turn.")
            return "".join(parts)
        return (
            f"{name} is NOT permitted for this request: "
            + "; ".join(self.reasons)
            + ". This cannot be fixed by retrying or rephrasing. Do not attempt it again; "
            "choose a different action or explain to the user why it cannot be done."
        )


@dataclass
class ReturnVerdict:
    """Verdict of ``controller.verify_return(action, result)`` on the return
    event of an executed call.

    A return event carries a result the environment has already produced,
    so it cannot be blocked; when recording it would falsify some contract's
    assumption, the enforcing controller *suppresses* it as a whole: none of
    its facts are written and the model does not receive the result.
    ``breakers`` names the contracts whose assumption the result would
    falsify; ``suppressed`` is True only on the enforcing side (an auditor
    records the break instead).
    """

    breakers: list[str] = field(default_factory=list)
    suppressed: bool = False

    def message(self, action: ProposedAction | None = None) -> str:
        """Feedback string fed back to the model in place of the result."""
        name = f"`{action.tool}`" if action else "the call"
        return (
            f"The result of {name} was withheld because it violates the environment "
            "assumption of: " + "; ".join(self.breakers) + ". "
            "Proceed without it, or explain to the user why the task cannot continue."
        )


class StepKind(str, Enum):
    NEURAL_CHOICE = "neural_choice"
    SYMBOLIC_EXECUTE = "symbolic_execute"
    DEAD_END = "dead_end"


@dataclass
class StepDecision:
    """Who controls the next step, per the Acuto runtime loop."""

    kind: StepKind
    actions: list[str] = field(default_factory=list)  # NEURAL_CHOICE: exposed tools
    action: ProposedAction | None = None  # SYMBOLIC_EXECUTE: the determined action
    reason: str = ""
    obligations: list["ActiveObligation"] = field(default_factory=list)

    @staticmethod
    def neural(
        actions: list[str], *, reason: str = "", obligations: list | None = None
    ) -> "StepDecision":
        return StepDecision(
            StepKind.NEURAL_CHOICE, actions=actions, reason=reason, obligations=obligations or []
        )

    @staticmethod
    def symbolic(action: ProposedAction, *, reason: str = "") -> "StepDecision":
        return StepDecision(StepKind.SYMBOLIC_EXECUTE, action=action, reason=reason)

    @staticmethod
    def dead_end(reason: str) -> "StepDecision":
        return StepDecision(StepKind.DEAD_END, reason=reason)
