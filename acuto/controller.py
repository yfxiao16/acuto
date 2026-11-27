"""``acuto.Controller``: one session of contract-based control, as an
object you can drive from any agent loop.

The controller exposes the stages of tool-call control as methods over
a :class:`~acuto.control.SymbolicController` and an optional tool registry:

* **mask** — :meth:`mask` returns the contract-admissible subset of the
  candidate tools; :meth:`route` additionally decides *who* acts next
  (fast-forward with no model call, the model over the masked tools, or
  a dead end).
* **route** — the :class:`~acuto.decisions.StepDecision` from
  :meth:`route`; a ``SYMBOLIC_EXECUTE`` decision carries the fully bound
  call the procedure determines.
* **verify** — :meth:`verify` is the hard boundary every call crosses
  before it executes, whether the model proposed it or the controller
  scheduled it. :meth:`commit` verifies the *return* event and advances
  the control state; it hands back what the model may see, which is a
  withheld-result error when the return would falsify an assumption.

:meth:`execute` chains verify → run the tool → commit for loops that let
Acuto run the tools; loops that execute tools elsewhere (a benchmark
harness, another framework's tool node) call :meth:`verify` before and
:meth:`commit` after. :func:`acuto.loop.iterate`, the loop behind
:class:`acuto.Agent`, is written against this same object, so a custom
loop and the built-in one share one code path.

    controller = acuto.Controller(library, tools=registry)
    while True:
        decision = controller.route(state=my_state)
        if decision.kind is acuto.StepKind.DEAD_END:
            break
        if decision.kind is acuto.StepKind.SYMBOLIC_EXECUTE:
            controller.execute(decision.action)          # no model call
            continue
        turn = my_model(messages, tools=registry.schemas(decision.actions))
        for call in turn.tool_calls:
            outcome = controller.execute((call.name, call.args))
            messages.append(outcome.feedback())           # result or refusal
    report = controller.finalize()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from acuto.contracts import CompiledContracts
from acuto.control import SymbolicController
from acuto.decisions import Decision, DecisionKind, ProposedAction, ReturnVerdict, StepDecision
from acuto.facts import PredicateContext
from acuto.tools import Tool, ToolExecutor, ToolRegistry, ToolResult
from acuto.tracing import Tracer


def as_action(call: Any) -> ProposedAction:
    """Coerce what a model adapter or a framework hands over into a
    :class:`ProposedAction`: a ``ProposedAction``, a ``(tool, args)``
    pair, a ``{"name": ..., "args": ...}`` dict, or any object with
    ``name``/``args`` (or ``arguments``) attributes such as a model
    :class:`~acuto.models.base.ToolCall`."""
    if isinstance(call, ProposedAction):
        return call
    if isinstance(call, tuple) and len(call) == 2:
        return ProposedAction(str(call[0]), dict(call[1] or {}))
    if isinstance(call, dict):
        name = call.get("name") or call.get("tool")
        args = call.get("args", call.get("arguments"))
        return ProposedAction(str(name), dict(args or {}))
    name = getattr(call, "name", None) or getattr(call, "tool", None)
    if name is None:
        raise TypeError(f"cannot read a tool call from {call!r}")
    args = getattr(call, "args", None)
    if args is None:
        args = getattr(call, "arguments", None)
    return ProposedAction(str(name), dict(args or {}))


@dataclass
class CallOutcome:
    """What :meth:`Controller.execute` did with one call.

    ``decision`` is the verify verdict. When it allowed the call,
    ``result`` is the tool's actual result and ``shown`` is what the
    model may see: the same result, or a withheld-result error when the
    return event was suppressed. When the call was blocked nothing ran
    and both are ``None``.
    """

    action: ProposedAction
    decision: Decision
    result: ToolResult | None = None
    shown: ToolResult | None = None
    return_verdict: ReturnVerdict | None = None

    @property
    def executed(self) -> bool:
        return self.result is not None

    @property
    def withheld(self) -> bool:
        return bool(self.return_verdict and self.return_verdict.suppressed)

    def feedback(self) -> dict:
        """The function-response payload for the model: the (possibly
        withheld) result when the call ran, the refusal otherwise."""
        if self.shown is not None:
            return self.shown.payload()
        return {"ok": False, "error": self.decision.message(self.action)}


class Controller:
    """One session of contract-based control over a tool-using agent.

    Build one per task (its control state is the session's), from a
    :class:`~acuto.library.ContractLibrary`, a list of contract specs, or
    compiled contracts. ``tools`` is optional: without a registry the
    controller still masks, routes, verifies, and commits, but cannot run
    tools itself (:meth:`execute` raises) and cannot bind fast-forward
    arguments from ``args_binder`` hooks.

    The keyword options are the ones :class:`acuto.Agent` takes:
    ``control_mode`` (``"full"`` | ``"mask"`` | ``"passive"``),
    ``mask_granularity`` (``"step"`` | ``"phase"`` | ``"hint"`` | ``"off"``),
    ``completion_check`` (also hide calls that leave the library jointly
    unsatisfiable), ``cache`` (a :class:`~acuto.cache.ResidualPolicyCache`
    shared across sessions, or ``True`` for a private one),
    ``predicate_evaluator`` (how ``fact(...)`` atoms are resolved),
    ``context_deriver`` (action-relative ``target_*`` context), and
    ``facts`` (initial facts, asserted before the first step).
    """

    def __init__(
        self,
        contracts: Any,
        *,
        tools: ToolRegistry | Iterable[Tool] | None = None,
        predicate_evaluator: Any = None,
        control_mode: str = "full",
        mask_granularity: str = "step",
        completion_check: bool = False,
        cache: Any = None,
        context_deriver: Callable[[ProposedAction, Any], dict] | None = None,
        facts: dict | None = None,
        tracer: Tracer | None = None,
        agent_id: str = "agent",
        pre_hooks: list[Callable[[ProposedAction], None]] | None = None,
        post_hooks: list[Callable[[ProposedAction, ToolResult], None]] | None = None,
    ) -> None:
        if isinstance(tools, ToolRegistry) or tools is None:
            self.registry = tools
        else:
            self.registry = ToolRegistry()
            for t in tools:
                self.registry.register(t)
        if cache is True:
            from acuto.cache import ResidualPolicyCache

            cache = ResidualPolicyCache()
        elif cache is False:
            cache = None
        compiled = (
            contracts.compiled
            if hasattr(contracts, "compiled")
            else contracts
            if isinstance(contracts, CompiledContracts)
            else list(contracts or [])
        )
        self.symbolic = SymbolicController(
            compiled,
            registry=self.registry,
            predicate_evaluator=predicate_evaluator,
            agent_id=agent_id,
            tracer=tracer,
            control_mode=control_mode,
            probe_cache=cache,
            mask_granularity=mask_granularity,
            completion_check=completion_check,
            context_deriver=context_deriver,
        )
        self.executor = (
            ToolExecutor(self.registry, pre_hooks=pre_hooks, post_hooks=post_hooks)
            if self.registry is not None
            else None
        )
        for predicate, value in (facts or {}).items():
            self.symbolic.assert_fact(predicate, value, source="<initial>")

    @classmethod
    def wrap(
        cls, symbolic: SymbolicController, *, tools: ToolRegistry | None = None
    ) -> "Controller":
        """A facade over an existing :class:`SymbolicController` (its
        state is shared, not copied)."""
        self = cls.__new__(cls)
        self.registry = tools if tools is not None else symbolic.registry
        if symbolic.registry is None and self.registry is not None:
            symbolic.registry = self.registry
        self.symbolic = symbolic
        self.executor = ToolExecutor(self.registry) if self.registry is not None else None
        return self

    # ------------------------------------------------------------------
    # Shared state
    # ------------------------------------------------------------------

    @property
    def facts(self):
        """The session's fact store (what tool results established)."""
        return self.symbolic.facts

    @property
    def obligations(self):
        return self.symbolic.obligations

    @property
    def tracer(self) -> Tracer:
        return self.symbolic.tracer

    @property
    def control_mode(self) -> str:
        return self.symbolic.control_mode

    @property
    def tool_names(self) -> list[str]:
        return self.registry.names() if self.registry is not None else []

    def assert_fact(self, predicate: str, value: Any = True, **metadata: Any) -> bool:
        """Record a fact from outside the tool boundary (the user's own
        words, an upstream system). Goes through the same assumption
        check as tool-result facts: under enforcement a write that would
        falsify an assumption is suppressed and ``False`` is returned."""
        metadata.setdefault("source", "<external>")
        return self.symbolic.assert_fact(predicate, value, **metadata)

    def invalidate_fact(self, predicate: str) -> None:
        self.symbolic.invalidate_fact(predicate)

    # ------------------------------------------------------------------
    # Stage 1 + 2: mask and route
    # ------------------------------------------------------------------

    def mask(self, candidates: Iterable[str] | None = None, state: Any = None) -> list[str]:
        """The contract-admissible subset of ``candidates`` (default: every
        registered tool) in the current control state. Argument-dependent
        contracts are not decidable here and are enforced by
        :meth:`verify` instead."""
        cands = list(candidates) if candidates is not None else self.tool_names
        return self.symbolic.admissible_actions(cands, state)

    def route(self, candidates: Iterable[str] | None = None, state: Any = None) -> StepDecision:
        """Who acts next. ``SYMBOLIC_EXECUTE`` carries the one call the
        procedure determines (execute it with no model call);
        ``NEURAL_CHOICE`` carries the tools to expose to the model;
        ``DEAD_END`` means no admissible call remains."""
        cands = list(candidates) if candidates is not None else self.tool_names
        return self.symbolic.next_decision(cands, state)

    def due_obligation(self, state: Any = None) -> ProposedAction | None:
        """The call a pending obligation still owes, with its arguments
        bound, or ``None``; a *next* obligation comes first. Loops consult
        this before letting a text-only turn end the session (the
        floor-yield gate)."""
        pending = self.obligations.due() + self.obligations.due_eventually()
        if not pending or self.control_mode != "full":
            return None
        ob = pending[0]
        args = self.obligations.resolve_args(ob, PredicateContext(self.facts, state))
        if args is None and self.registry is not None and ob.tool in self.registry:
            if not self.registry.get(ob.tool).required_params:
                args = {}
        return ProposedAction(ob.tool, args) if args is not None else None

    # ------------------------------------------------------------------
    # Stage 3: verify, execute, commit
    # ------------------------------------------------------------------

    def verify(self, call: Any, state: Any = None) -> Decision:
        """Is this concrete call executable now? All contracts, real
        arguments. ``ALLOW`` executes; ``REQUIRE`` names what is missing
        and which tools establish it; ``BLOCK`` is final."""
        return self.symbolic.validate(as_action(call), state)

    def verify_batch(self, calls: Iterable[Any], state: Any = None) -> list[Decision]:
        """Verify calls the model emitted in one message, each against the
        state the admitted calls before it would leave."""
        return self.symbolic.validate_batch([as_action(c) for c in calls], state)

    def commit(
        self, call: Any, result: ToolResult, state: Any = None, *, step: int | None = None
    ) -> ToolResult:
        """Commit an executed call and verify its return event.

        The call event always commits (it happened). The return is checked
        against every assumption as a whole: when recording it would
        falsify one, the enforcing controller suppresses it, writes no
        facts from it, and the returned :class:`ToolResult` is a
        withheld-result error to show the model in place of the result.
        Otherwise the result is returned unchanged.
        """
        action = as_action(call)
        verdict = self.symbolic.verify_return(action, result, state)
        self.symbolic.update(action, result, state, return_verdict=verdict)
        if not verdict.suppressed:
            return result
        rec: dict[str, Any] = {"tool": action.tool, "contracts": list(verdict.breakers)}
        if step is not None:
            rec["step"] = step
        self.tracer.record("return/withheld", **rec)
        return ToolResult(
            tool=action.tool, args=dict(action.args), ok=False, error=verdict.message(action)
        )

    def execute(self, call: Any, state: Any = None, *, step: int | None = None) -> CallOutcome:
        """verify → run the tool → commit, as one step. Needs ``tools``."""
        if self.executor is None:
            raise RuntimeError(
                "Controller.execute needs a tool registry; pass tools=... or use "
                "verify()/commit() around your own execution"
            )
        action = as_action(call)
        decision = self.verify(action, state)
        if decision.kind is not DecisionKind.ALLOW:
            return CallOutcome(action=action, decision=decision)
        result = self.executor.execute(action)
        verdict = self.symbolic.verify_return(action, result, state)
        self.symbolic.update(action, result, state, return_verdict=verdict)
        shown = result
        if verdict.suppressed:
            rec: dict[str, Any] = {"tool": action.tool, "contracts": list(verdict.breakers)}
            if step is not None:
                rec["step"] = step
            self.tracer.record("return/withheld", **rec)
            shown = ToolResult(
                tool=action.tool, args=dict(action.args), ok=False, error=verdict.message(action)
            )
        return CallOutcome(
            action=action, decision=decision, result=result, shown=shown, return_verdict=verdict
        )

    # ------------------------------------------------------------------
    # Session end
    # ------------------------------------------------------------------

    def finalize(self) -> dict:
        """End-of-session report: contracts left violated at the end,
        obligations still pending, contracts whose assumption broke."""
        return self.symbolic.finalize()
