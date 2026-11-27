"""The Symbolic Controller — Acuto's central abstraction.

Owns the symbolic control state (LTLf residuals + facts + obligations +
grounding accumulators) and answers four questions:

* ``admissible_actions``  — which tools may be exposed to the model now?
  (cheap, tool-level, argument-dependent contracts excluded)
* ``validate``            — is this concrete (tool, args) executable now?
  (the hard pre-execution boundary; all contracts, real arguments)
* ``next_decision``       — who controls the next step: symbolic
  execution, neural choice over a narrowed toolset, or dead end?
* ``update``              — commit an executed action into both the
  trace monitors and the fact store (via result extractors).

Filtering and validation are deliberately separate APIs even though v0
shares most logic between them: future versions keep filtering cheap
while validate() grows richer checks (argument constraints, evidence,
external predicates).

Pipeline (the DFA never inspects application data directly):

    environment / agent state
        → PredicateEvaluator → fact propositions
        → grounding(event)   → trace propositions
        → contract progression (LTLfBackend)
        → control state → admissible actions
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import json
from typing import Any, Callable

from contragent import Event
from contragent.formulas import And, Atom, Or
from contragent.kernel import GroundingState, ground_event

from acuto.backend import LTLfBackend
from acuto.contracts import (
    ACTION_CTX_PREFIX,
    CompiledContracts,
    Contract,
    CustomRule,
    compile_contracts,
)
from acuto.decisions import Decision, DecisionKind, ProposedAction, ReturnVerdict, StepDecision
from acuto.facts import FactStore, LocalPredicateEvaluator, PredicateContext, PredicateEvaluator
from acuto.obligations import ObligationEngine
from acuto.tools import ToolRegistry, ToolResult
from acuto.tracing import Tracer


def _canonical(node: Any) -> Any:
    """A residual up to associativity, commutativity and idempotence of
    ``&`` and ``|``. The DFA evaluator does not normalize these, so a
    residual can grow while its language stays the same (``G p`` stepped
    on an idle event becomes ``G p & G p``)."""
    if isinstance(node, (And, Or)):
        kind, parts, stack = type(node), set(), [node]
        while stack:
            n = stack.pop()
            if isinstance(n, kind):
                stack += [n.left, n.right]
            else:
                parts.add(_canonical(n))
        return (kind.__name__, frozenset(parts))
    children = [getattr(node, a) for a in ("child", "left", "right") if hasattr(node, a)]
    if children:
        return (type(node).__name__, *(_canonical(c) for c in children))
    return node


def _stable(node: Any) -> str:
    """:func:`_canonical` as a string that is the same in every process
    (a frozenset's repr follows string hashing, which is randomized)."""
    if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], frozenset):
        return f"{node[0]}{{{', '.join(sorted(_stable(x) for x in node[1]))}}}"
    if isinstance(node, tuple):
        return f"{node[0]}({', '.join(_stable(x) for x in node[1:])})"
    return repr(node)


# A failed call was attempted but did nothing: it counts toward limits and
# loop detection (the counters still advance) but fulfils no obligation, so
# a "log out" whose logout call failed is still owed.
_CALL_ATOMS = ("called(", "called_with(", "called_any")


def _attempt_only(valuation: dict) -> dict:
    return {k: v for k, v in valuation.items() if not k.startswith(_CALL_ATOMS)}


def _journaled(encode: Callable[..., dict]):
    """Record a call of the wrapped input method in the controller's
    journal, unless it is nested in another journaled call (the facts an
    ``update`` writes are replayed by replaying the ``update``)."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if self._journal_depth:
                return fn(self, *args, **kwargs)
            entry = encode(*args, **kwargs)
            self._journal_depth += 1
            try:
                return fn(self, *args, **kwargs)
            finally:
                self._journal_depth -= 1
                self.journal.append(entry)

        return wrapper

    return deco


def _enc_update(action, result, agent_state=None, **_):
    return {
        "op": "update",
        "tool": action.tool,
        "args": copy.deepcopy(dict(action.args)),
        "result": {
            "tool": result.tool,
            "args": copy.deepcopy(dict(result.args)),
            "ok": result.ok,
            "output": copy.deepcopy(result.output),
            "error": result.error,
        },
        "state": agent_state,
    }


def _enc_assert(predicate, value=True, *, agent_state=None, checked=False, **metadata):
    return {
        "op": "assert_fact",
        "predicate": predicate,
        "value": copy.deepcopy(value),
        "state": agent_state,
        "checked": checked,
        "metadata": dict(metadata),
    }


def _enc_invalidate(predicate):
    return {"op": "invalidate_fact", "predicate": predicate}


def _enc_idle(agent_state=None):
    return {"op": "idle", "state": agent_state}


def _enc_observe(tool, agent_state=None, *, args=None):
    out = {"op": "observe_call", "tool": tool, "state": agent_state}
    if args is not None:
        out["args"] = dict(args)
    return out


def _formula_text(f):
    return f if f is None or isinstance(f, str) else repr(f)


def _enc_attach(rule_id, formula, assumption=None, *, name, advisory=True, meta=None):
    return {
        "op": "attach_rule",
        "rule": rule_id,
        "formula": _formula_text(formula),
        "assumption": _formula_text(assumption),
        "name": name,
        "advisory": advisory,
        "meta": copy.deepcopy(meta or {}),
    }


def _enc_confirm(rule_id):
    return {"op": "confirm_rule", "rule": rule_id}


def _enc_detach(rule_id):
    return {"op": "detach_rule", "rule": rule_id}


def _enc_own(name, owner):
    return {"op": "set_owner", "contract": name, "owner": owner}


def _enc_suspend(name, by, *, until=None, quote=""):
    return {"op": "suspend", "contract": name, "by": by, "until": until, "quote": quote}


def _enc_release(by):
    return {"op": "release", "by": by}


def _enc_end_request():
    return {"op": "end_request"}


class SymbolicController:
    def __init__(
        self,
        contracts: CompiledContracts | list,
        *,
        registry: ToolRegistry | None = None,
        predicate_evaluator: PredicateEvaluator | None = None,
        agent_id: str = "agent",
        tracer: Tracer | None = None,
        control_mode: str = "full",
        probe_cache=None,
        mask_granularity: str = "step",
        completion_check: bool = False,
        enforcing: bool = True,
        context_deriver: Callable[[ProposedAction, FactStore], dict] | None = None,
    ) -> None:
        # enforcing=False is observe mode: the controller records what a run did
        # and never intervenes. An auditor runs this way, so that replaying ANY
        # condition's trace — including one whose environment broke an
        # assumption — yields a faithful record rather than a record of what
        # enforcement would have prevented. Suppression is an enforcement
        # action, so it belongs only to the enforcing side.
        self.enforcing = enforcing
        # Idle (a step with no call) as one of the agent's actions: the
        # jump-forward then fires only when idling would leave a guarantee
        # unsatisfiable. Off by default, where a single admissible call is
        # taken as the step to make.
        self.idle_is_action = False
        # Contracts whose open eventualities keep the agent from stopping
        # (a subtask's deliverable): the run loop nudges as for owed rules.
        self.deliverables: set[str] = set()
        # Tools whose call ends the run (AppWorld's complete_task): nothing
        # can come after them, so one that would leave a binding contract
        # owing an eventuality is blocked.
        self.terminal_tools: set[str] = set()
        # Off by default: the one-step mask hides a tool whose call violates a
        # contract now. With completion_check=True each surviving tool also pays a
        # joint satisfiability query, which additionally hides a call that
        # leaves the library jointly unsatisfiable (a dead end). Libraries
        # that pass contragent.analysis.dead_ends.check_dead_ends cannot block,
        # so the query is pure overhead for them.
        self.completion_check = completion_check
        self._coacc_cache: dict = {}
        # "step": expose only the per-step admissible set (max masking, but
        # schema churn breaks provider prompt caches). "phase": expose the
        # flow's stable candidate set; contract admissibility still drives
        # jump detection internally and validate() enforces per call —
        # trading some invalid-proposal risk for prefix-cache stability.
        # "hint" as "phase", plus a message naming the admissible tools.
        # "off": no guidance at all, validate() only (an ablation).
        if mask_granularity not in ("step", "phase", "hint", "off"):
            raise ValueError(f"unknown mask_granularity {mask_granularity!r}")
        self.mask_granularity = mask_granularity
        self.probe_cache = probe_cache
        # control_mode — the ablation ladder:
        #   "passive": validate-only (block + reprompt); no masking, no jump.
        #   "mask":    dynamic tool exposure, but no symbolic execution
        #              (a singleton/obligation still goes to the LLM).
        #   "full":    mask + symbolic jump-forward (Acuto).
        if control_mode not in ("full", "mask", "passive"):
            raise ValueError(f"unknown control_mode {control_mode!r}")
        self.control_mode = control_mode
        cc = contracts if isinstance(contracts, CompiledContracts) else compile_contracts(contracts)
        self.contracts = cc
        self.backend = LTLfBackend(cc.contracts)
        self.facts = FactStore()
        self.obligations = ObligationEngine(cc.obligations)
        self.registry = registry
        self.agent_id = agent_id
        self.tracer = tracer or Tracer()
        self._pe = predicate_evaluator or LocalPredicateEvaluator()
        self._grounding = GroundingState()
        self.events: list[Event] = []
        # One entry per committed event, aligned with ``events``: the tool
        # called (None for an idle event) and the facts its valuation saw.
        # A caller outside the run may evaluate preconditions over it.
        self.fact_history: list[tuple[str | None, dict[str, Any]]] = []
        # Every input that moved this controller, in order: calls with their
        # results, idle events, replayed foreign calls, fact writes from
        # outside. A checkpoint carries it; replaying it into a fresh
        # controller (:meth:`replay`) rebuilds residuals, facts and
        # obligations, under the same contracts or changed ones.
        self.journal: list[dict] = []
        self._journal_depth = 0
        # Rules attached mid-run (acuto.rules): id -> name, status, the
        # event index of each change, and the provenance the caller gave.
        self.rules: dict[str, dict] = {}
        # Contracts the user's later words set aside (see suspend): name ->
        # {"contract", "by": set of suspender ids, "at"}; and suspender id ->
        # {"until": None | "request", "quote"}.
        self.suspended: dict[str, dict] = {}
        self.suspenders: dict[str, dict] = {}
        # Action-relative context (the ``target_*`` convention): facts about
        # the entity a proposed action names, computed from the base facts
        # at the moment that action is judged, and present in the grounding
        # only while it is probed or stepped. This is the η seam for rules
        # like "cancel only an eligible reservation", whose truth depends on
        # which reservation the call names — a binding LTLf cannot express.
        self._context_deriver = context_deriver

    def _probe_ctx(self, tool: str) -> dict[str, str]:
        """The deriver's context for ``tool`` with no arguments yet, minus
        the argument-dependent keys; empty if the deriver needs arguments."""
        try:
            derived = self._derived_ctx(ProposedAction(tool, {}))
        except Exception:  # noqa: BLE001 - a deriver may need the arguments
            return {}
        return {k: v for k, v in derived.items() if not k.startswith(ACTION_CTX_PREFIX)}

    def _derived_ctx(self, action: ProposedAction) -> dict[str, str]:
        if self._context_deriver is None:
            return {}
        derived = self._context_deriver(action, self.facts) or {}
        return {str(k): str(v) for k, v in derived.items()}

    # ------------------------------------------------------------------
    # Facts (all writes flow through here so obligations + tracing fire)
    # ------------------------------------------------------------------

    @_journaled(_enc_assert)
    def assert_fact(
        self,
        predicate: str,
        value: Any = True,
        *,
        agent_state: Any = None,
        checked: bool = False,
        **metadata: Any,
    ) -> bool:
        """Write one fact. Returns False if the write was suppressed.

        ``checked=True`` skips the per-write assumption probe: the caller has
        already verified the whole return event this write belongs to
        (:meth:`verify_return`), so the write cannot break an assumption once
        its siblings land, even if it would on its own.

        This is the single funnel for every environment-side write into the
        control state, so it is also where Acuto maintains the assumption
        side of its contracts. A fact reaches ``κ`` from a declared result
        extractor or from the initial task facts — the environment, never
        the agent — so a write that would falsify some contract's
        assumption is the environment event ContrAgent suppresses. Acuto
        suppresses it here: the fact is not recorded, the control state does
        not advance on it, and the contract stays ACTIVE instead of falling
        IDLE and silently discharging its guarantee.

        Only an enforcing controller suppresses. In observe mode the write
        lands and the break is traced, so an audit reports what the run
        actually did (see ``enforcing`` in :meth:`__init__`)."""
        breakers = [] if checked else self._assumption_breakers(predicate, value, agent_state)
        if breakers and self.enforcing:
            self.tracer.record(
                "fact/suppressed",
                predicate=predicate,
                value=value,
                contracts=[c.name for c in breakers],
            )
            return False
        if breakers:
            # Observe mode: the environment did break its side, and an auditor's
            # job is to say so, not to rewrite history.
            self.tracer.record(
                "fact/broke_assumption",
                predicate=predicate,
                value=value,
                contracts=[c.name for c in breakers],
            )
        fact = self.facts.assert_fact(predicate, value, **metadata)
        self.tracer.record(
            "fact/asserted", predicate=predicate, value=value, metadata=fact.metadata
        )
        if fact.truthy:
            for ob in self.obligations.on_fact_asserted(predicate, step=len(self.events)):
                self.tracer.record("obligation/created", obligation=ob.desc, tool=ob.tool)
        return True

    def _assumption_breakers(self, predicate: str, value: Any, agent_state: Any) -> list[Contract]:
        """Contracts whose assumption this fact write would falsify.

        Costs nothing when no loaded contract carries an assumption, which
        is every library shipped with Acuto today."""
        if not self.contracts.has_assumptions:
            return []
        probe = self.facts.checkpoint()
        try:
            self.facts.assert_fact(predicate, value)
            valuation = self._fact_valuation(agent_state)
        finally:
            self.facts.restore(probe)
        return self.backend.assumption_breakers(valuation)

    @_journaled(_enc_invalidate)
    def invalidate_fact(self, predicate: str) -> None:
        retracted = self.facts.invalidate_fact(predicate)
        if retracted is not None:
            self.tracer.record("fact/invalidated", predicate=predicate)

    # ------------------------------------------------------------------
    # Valuation construction
    # ------------------------------------------------------------------

    def _fact_valuation(self, agent_state: Any = None) -> dict[str, object]:
        """Resolve every contract-referenced fact predicate to a proposition."""
        ctx = PredicateContext(self.facts, agent_state)
        out: dict[str, object] = {}
        for p in sorted(self.contracts.fact_predicates):
            val = self._pe.evaluate(p, ctx)
            out[Atom("fact", p).key()] = bool(val)
            if val is not None and not isinstance(val, bool) and isinstance(val, (int, float)):
                out[Atom("fact_value", p).key()] = val
        return out

    def _valuation_for(
        self,
        tool: str,
        args: dict | None,
        agent_state: Any,
        grounding: GroundingState,
    ) -> dict[str, object]:
        event = Event(
            ts=len(self.events),
            agent=self.agent_id,
            event_type="tool_call",
            tool=tool,
            args=dict(args or {}),
        )
        valuation = ground_event(event, len(self.events), grounding, self.contracts.content_atoms)
        valuation.update(self._fact_valuation(agent_state))
        return valuation

    # ------------------------------------------------------------------
    # Action-space compilation (cheap, tool-level)
    # ------------------------------------------------------------------

    def _cache_key_base(self, agent_state: Any):
        """Structural part of the residual-policy cache key (no session
        values — only contract-referenced predicate valuations)."""
        g = self._grounding
        facts_fp = tuple(sorted(self._fact_valuation(agent_state).items()))
        return (
            self.backend.state_signature(),
            tuple(sorted(g.call_counts.items())),
            tuple(sorted(g.consecutive_counts.items())),
            g.last_tool,
            facts_fp,
        )

    def admissible_actions(
        self, candidate_actions: list[str], agent_state: Any = None
    ) -> list[str]:
        admissible = []
        key_base = self._cache_key_base(agent_state) if self.probe_cache is not None else None
        for tool in candidate_actions:
            # Context the deriver gives this tool before any arguments
            # (keys that depend on the state only, like "is the work
            # done?"); without it a rule over such a key read it as unset
            # and hid the tool for good. Argument-dependent keys
            # (``target_*``) are judged at validate, not here.
            derived = self._probe_ctx(tool)
            key = (key_base, tool, tuple(sorted(derived.items())))
            violated_names = None
            if key_base is not None:
                violated_names = self.probe_cache.lookup(key)
            if violated_names is None:
                grounding = copy.deepcopy(self._grounding)
                grounding.current_ctx.update(derived)
                valuation = self._valuation_for(tool, {}, agent_state, grounding)
                violated = self.backend.probe(
                    valuation, include_args_dependent=False, masking_only=True
                )
                violated_names = tuple(r.name for r in violated)
                if key_base is not None:
                    self.probe_cache.store(key, violated_names)
            if violated_names:
                self.tracer.record("action/masked", tool=tool, contracts=list(violated_names))
                continue
            if self.completion_check and not self._completable_after(tool, agent_state, key_base):
                self.tracer.record("action/masked", tool=tool, contracts=["<dead-end>"])
                continue
            admissible.append(tool)
        return admissible

    def _completable_after(self, tool: str, agent_state: Any, key_base) -> bool:
        """Would calling ``tool`` leave the library jointly satisfiable?

        The one-step mask asks each contract on its own, so a sequence of
        individually legal calls can still reach a state no continuation
        completes. This asks the joint question on the residuals the call
        would leave behind. An inconclusive query counts as admissible: the
        mask may not hide a tool that has a legal call.
        """
        if key_base is not None:
            cached = self._coacc_cache.get((key_base, tool))
            if cached is not None:
                return cached
        from contragent.kernel import completable

        grounding = copy.deepcopy(self._grounding)
        valuation = self._valuation_for(tool, {}, agent_state, grounding)
        residuals = self.backend.residuals_after(valuation)
        counters = dict(grounding.call_counts)
        verdict = completable(residuals, counters=counters)
        ok = verdict is not False
        if key_base is not None:
            self._coacc_cache[(key_base, tool)] = ok
        return ok

    def _completable(self, valuation: dict, grounding) -> bool:
        """The mask's joint question, asked with the arguments in hand.

        Same query as :meth:`_completable_after`, at full precision: the
        arguments are known, so every contract of the library takes part,
        not only the tool-level ones. One query per step rather than one
        per candidate. An inconclusive query counts as completable.
        """
        from contragent.kernel import completable

        residuals = self.backend.residuals_after(
            valuation, include_args_dependent=True, masking_only=False
        )
        verdict = completable(residuals, counters=dict(grounding.call_counts))
        return verdict is not False

    # ------------------------------------------------------------------
    # Hard pre-execution validation (all contracts, concrete arguments)
    # ------------------------------------------------------------------

    def validate(self, action: ProposedAction, agent_state: Any = None) -> Decision:
        grounding = copy.deepcopy(self._grounding)
        grounding.current_ctx.update(self._derived_ctx(action))
        valuation = self._valuation_for(action.tool, action.args, agent_state, grounding)
        violated = self.backend.probe(valuation, include_args_dependent=True, include_advisory=True)
        warnings = [c.name for c in violated if c.advisory]
        violated = [c for c in violated if not c.advisory]
        if not violated:
            if self.completion_check and not self._completable(valuation, grounding):
                return Decision(DecisionKind.BLOCK, reasons=["<dead-end>"])
            if action.tool in self.terminal_tools:
                owed = self._owed_after(valuation)
                if owed:
                    return Decision(
                        DecisionKind.BLOCK,
                        reasons=[f"{n} (still owed; {action.tool} ends the run)" for n in owed],
                    )
            return Decision(DecisionKind.ALLOW, warnings=warnings)
        if all(contract.recoverable for contract in violated):
            return self._require_decision(violated, agent_state)
        hard = [r for r in violated if not r.recoverable]
        return Decision(DecisionKind.BLOCK, reasons=[r.name for r in hard])

    def validate_batch(
        self, actions: list[ProposedAction], agent_state: Any = None
    ) -> list[Decision]:
        """Validate calls the model emitted together in ONE message.

        The formal model is one action per step and the environment runs a
        batch in order, so call k is admissible iff it is admissible in the
        state reached after the admitted calls 1..k-1 — not, as a per-call
        :meth:`validate` against the pre-batch state says, iff each call is
        admissible on its own. Per-call validation let one message carrying
        several ``cancel_reservation`` calls pass an "at most 2" bound
        because each call saw a count of zero.

        Nothing is committed. The admitted prefix is stepped through the
        monitors and counters speculatively — as a successful call with no
        result, so no facts, obligations or trace records — and the state is
        restored on exit. A single call is exactly :meth:`validate`.
        """
        if len(actions) <= 1:
            return [self.validate(a, agent_state) for a in actions]
        saved_backend = self.backend.snapshot()
        saved_grounding = copy.deepcopy(self._grounding)
        n_events = len(self.events)
        try:
            decisions: list[Decision] = []
            for action in actions:
                decision = self.validate(action, agent_state)
                decisions.append(decision)
                if decision.allowed:
                    self._speculative_step(action, agent_state)
            return decisions
        finally:
            self.backend.restore(saved_backend)
            self._grounding = saved_grounding
            del self.events[n_events:]

    def _speculative_step(self, action: ProposedAction, agent_state: Any = None) -> None:
        """Advance monitors and counters as if ``action`` had just succeeded.

        The first half of :meth:`update` only: with no result there is nothing
        to extract, and the caller restores everything afterwards."""
        args = dict(action.args)
        args["succeeded"] = 1
        event = Event(
            ts=len(self.events),
            agent=self.agent_id,
            event_type="tool_call",
            tool=action.tool,
            args=args,
        )
        derived = self._derived_ctx(action)
        self._grounding.current_ctx.update(derived)
        try:
            valuation = ground_event(
                event, len(self.events), self._grounding, self.contracts.content_atoms
            )
            valuation.update(self._fact_valuation(agent_state))
            self.backend.step(valuation)
        finally:
            for k in derived:
                self._grounding.current_ctx.pop(k, None)
        self.events.append(event)

    def _require_decision(self, violated: list[Contract], agent_state: Any = None) -> Decision:
        reasons, requirements, hints = [], [], []
        ctx = PredicateContext(self.facts, agent_state)
        for contract in violated:
            reasons.append(contract.name)
            for predicate in contract.requirements:
                # Same resolution the backend sees (derived predicates go
                # through the evaluator, not the raw store).
                if bool(self._pe.evaluate(predicate, ctx)):
                    continue
                requirements.append(predicate)
                hints.extend(self._hints_for(predicate, ctx))
            for tool in contract.prerequisites:
                requirements.append(f"called({tool})")
                hints.append(f"call `{tool}`")
        seen: set[str] = set()
        hints = [h for h in hints if not (h in seen or seen.add(h))]
        return Decision(
            DecisionKind.REQUIRE, reasons=reasons, requirements=requirements, hints=hints
        )

    def _hints_for(self, predicate: str, ctx: PredicateContext) -> list[str]:
        """Recovery hints for one unmet predicate.

        A directly asserted predicate names the tool that asserts it. A
        derived predicate (computed by the evaluator, asserted by no tool)
        is expanded through ``evaluator.missing`` to the base facts it
        still lacks, each naming its tool. When nothing resolves to a tool
        the hint says so outright: a predicate name left bare in the
        feedback gets proposed as a tool call (traced 2026-09: Sonnet
        called ``report_ready`` 56 times in one aircraft run).
        """
        tools = self.contracts.asserted_by.get(predicate, [])
        if tools:
            return [f"call `{tool}` to establish {predicate}" for tool in tools]
        missing_fn = getattr(self._pe, "missing", None)
        base = missing_fn(predicate, ctx) if callable(missing_fn) else None
        hints = [
            f"call `{tool}` to establish {b} (needed for {predicate})"
            for b in (base or [])
            for tool in self.contracts.asserted_by.get(b, [])
        ]
        if hints:
            return hints
        return [
            f"establish {predicate} by completing the remaining required steps "
            f"({predicate} is a condition, not a tool: never call it)"
        ]

    # ------------------------------------------------------------------
    # Control handoff
    # ------------------------------------------------------------------

    def state_signature(self) -> int:
        """Control-state signature for residual-policy-cache research:
        (residuals, facts, call counters) — see LTLfBackend.state_signature."""
        facts_sig = tuple(sorted((k, str(f.value)) for k, f in self.facts._facts.items()))
        counters = tuple(sorted(self._grounding.call_counts.items()))
        try:
            return hash((self.backend.state_signature(), facts_sig, counters))
        except TypeError:  # unhashable residual leaf — degrade to repr
            return hash((repr(self.backend.state_signature()), facts_sig, counters))

    def next_decision(self, candidate_actions: list[str], agent_state: Any = None) -> StepDecision:
        self.tracer.record("controller/state_sig", sig=self.state_signature())
        # Passive verifier: no action-space shaping at all — the hard
        # validate() boundary is the only control (block + reprompt).
        if self.control_mode == "passive":
            return StepDecision.neural(
                list(candidate_actions), obligations=self.obligations.pending()
            )

        # Case C — an obligation is due: symbolic execution preempts choice.
        due = self.obligations.due()
        if due and self.control_mode == "mask":
            # Mask-only: the obligation narrows the toolset but the LLM
            # still makes (and argues) the call itself.
            return StepDecision.neural(
                [due[0].tool], reason=f"obligation (mask-only): {due[0].desc}", obligations=due
            )
        if due:
            ob = due[0]
            ctx = PredicateContext(self.facts, agent_state)
            args = self.obligations.resolve_args(ob, ctx)
            if args is None:
                args = self._bind_trivial_args(ob.tool)
            if args is not None:
                return StepDecision.symbolic(
                    ProposedAction(ob.tool, args), reason=f"obligation: {ob.desc}"
                )
            # Arguments need interpretation → the LLM decides, but only
            # over the obligated action (singleton action space).
            return StepDecision.neural(
                [ob.tool], reason=f"obligation requires arguments: {ob.desc}", obligations=due
            )

        admissible = self.admissible_actions(candidate_actions, agent_state)
        if not admissible:
            return StepDecision.dead_end("no admissible actions remain")

        # Case B — exactly one admissible action whose arguments are
        # procedurally determined: jump-forward, no model call. When idle
        # counts as an action, the call must also be the only live one.
        if (
            len(admissible) == 1
            and self.control_mode == "full"
            and not (self.idle_is_action and self.live_after(None, agent_state))
        ):
            args = self._bind_args(admissible[0], agent_state)
            if args is not None:
                return StepDecision.symbolic(
                    ProposedAction(admissible[0], args),
                    reason="single admissible action with determined arguments",
                )

        # Case A — genuine freedom: the model chooses among admissible actions.
        self.last_admissible = list(admissible)
        exposed = (
            list(candidate_actions)
            if self.mask_granularity in ("phase", "hint", "off")
            else admissible
        )
        return StepDecision.neural(exposed, obligations=self.obligations.pending())

    def _bind_args(self, tool_name: str, agent_state: Any = None) -> dict | None:
        """Deterministic argument binding; None = LLM must decide."""
        if self.registry is None or tool_name not in self.registry:
            return None
        tool = self.registry.get(tool_name)
        if tool.args_binder is not None:
            return tool.args_binder(PredicateContext(self.facts, agent_state))
        if not tool.required_params:
            return {}
        return None

    def _bind_trivial_args(self, tool_name: str) -> dict | None:
        if self.registry is not None and tool_name in self.registry:
            if not self.registry.get(tool_name).required_params:
                return {}
        return None

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def _return_writes(
        self, action: ProposedAction, result: ToolResult
    ) -> tuple[list[tuple[str, Any, dict]], list[str]]:
        """The fact writes and invalidations the return event of ``action``
        would make: what the declared extractors read off ``result``. A call
        that returned an error writes only the assertions made ``on_error``
        and invalidates nothing."""
        writes: list[tuple[str, Any, dict]] = []
        for spec in self.contracts.extractors.get(action.tool, []):
            if not result.ok and not spec.on_error:
                continue
            if spec.when is not None and not spec.when(result):
                continue
            value = spec.value(result) if callable(spec.value) else spec.value
            metadata = spec.metadata(result) if callable(spec.metadata) else (spec.metadata or {})
            name = spec.predicate(result) if callable(spec.predicate) else spec.predicate
            writes.append((name, value, dict(metadata)))
        invalidations = list(self.contracts.invalidations.get(action.tool, [])) if result.ok else []
        return writes, invalidations

    def verify_return(
        self, action: ProposedAction, result: ToolResult, agent_state: Any = None
    ) -> ReturnVerdict:
        """The enforcement map on the return event $(T, result)$.

        The call has executed, so the event cannot be blocked; what Acuto
        decides is whether the agent's control state may advance on it.
        All facts the result would write are applied on a checkpoint and
        the joint valuation is checked against every assumption, so the
        verdict is on the return event as a whole, not per fact. An
        enforcing controller suppresses a breaking return (:meth:`update`
        then writes nothing from it and the caller withholds the result
        from the model); an auditor only reports the breakers.
        """
        if not self.contracts.has_assumptions:
            return ReturnVerdict()
        writes, invalidations = self._return_writes(action, result)
        if not writes and not invalidations:
            return ReturnVerdict()
        probe = self.facts.checkpoint()
        try:
            for name, value, _ in writes:
                self.facts.assert_fact(name, value)
            for predicate in invalidations:
                self.facts.invalidate_fact(predicate)
            valuation = self._fact_valuation(agent_state)
        finally:
            self.facts.restore(probe)
        breakers = [c.name for c in self.backend.assumption_breakers(valuation)]
        return ReturnVerdict(breakers=breakers, suppressed=bool(breakers) and self.enforcing)

    @_journaled(_enc_update)
    def update(
        self,
        action: ProposedAction,
        result: ToolResult,
        agent_state: Any = None,
        *,
        return_verdict: ReturnVerdict | None = None,
    ) -> None:
        """Commit an executed action into both worlds: trace monitors first,
        then fact extraction from the structured ToolResult.

        The call event always commits (it happened). The return event
        commits only if :meth:`verify_return` passes it: a suppressed
        return writes no facts and retracts nothing, so the control state
        does not advance on a result the model never receives. Pass the
        verdict in when the caller already computed it to decide what to
        show the model; otherwise it is computed here."""
        if return_verdict is None:
            return_verdict = self.verify_return(action, result, agent_state)
        args = dict(action.args)
        args["succeeded"] = 1 if result.ok else 0
        event = Event(
            ts=len(self.events),
            agent=self.agent_id,
            event_type="tool_call",
            tool=action.tool,
            args=args,
        )
        derived = self._derived_ctx(action)
        self._grounding.current_ctx.update(derived)
        self.fact_history.append((action.tool, self.facts.snapshot()))
        try:
            valuation = ground_event(
                event, len(self.events), self._grounding, self.contracts.content_atoms
            )
            valuation.update(self._fact_valuation(agent_state))
            if not result.ok:
                valuation = _attempt_only(valuation)
            violated = self.backend.step(valuation)
        finally:
            for k in derived:
                self._grounding.current_ctx.pop(k, None)
        self.events.append(event)
        # A hard violation is only reachable in observe-style flows
        # (enforce blocks first); an advisory one is allowed through.
        self._record_violations(action.tool, violated)

        if return_verdict.suppressed:
            self.tracer.record(
                "return/suppressed", tool=action.tool, contracts=list(return_verdict.breakers)
            )
        else:
            writes, invalidations = self._return_writes(action, result)
            # A verdict with no breakers has cleared the joint write, so the
            # per-write probe is skipped; an auditor's verdict with breakers
            # falls through to it, which records the break without suppressing.
            checked = not return_verdict.breakers
            for name, value, metadata in writes:
                self.assert_fact(
                    name,
                    value,
                    agent_state=agent_state,
                    checked=checked,
                    source=action.tool,
                    **metadata,
                )
            for predicate in invalidations:
                self.invalidate_fact(predicate)

        for ob in self.obligations.on_action_executed(action.tool, result.ok):
            self.tracer.record("obligation/satisfied", obligation=ob.desc, tool=ob.tool)

    # ------------------------------------------------------------------
    # Idle events and liveness
    # ------------------------------------------------------------------

    def _idle_event(self) -> Event:
        # A "message" event grounds no call atom: the valuation is the facts.
        return Event(ts=len(self.events), agent=self.agent_id, event_type="message")

    def _idle_valuation(self, agent_state: Any, grounding: GroundingState) -> dict[str, object]:
        valuation = ground_event(
            self._idle_event(), len(self.events), grounding, self.contracts.content_atoms
        )
        valuation.update(self._fact_valuation(agent_state))
        return valuation

    @_journaled(_enc_idle)
    def idle(self, agent_state: Any = None) -> list[Contract]:
        """Commit an event with no tool call; the monitors advance on the
        current facts. A model whose policy is to wait produces one. An
        idle event cannot
        be blocked, so a guarantee that demanded a call now is violated
        here and traced like any committed violation."""
        self.fact_history.append((None, self.facts.snapshot()))
        # Stopping is judged on the state as it is: the state-only context
        # (as for a mask probe) is derived for the idle event too.
        derived = self._probe_ctx("")
        self._grounding.current_ctx.update(derived)
        try:
            valuation = self._idle_valuation(agent_state, self._grounding)
            violated = self.backend.step(valuation)
        finally:
            for k in derived:
                self._grounding.current_ctx.pop(k, None)
        self.events.append(self._idle_event())
        return self._record_violations(None, violated)

    def _owed_after(self, valuation: dict[str, object]) -> list[str]:
        """Binding contracts that would fail if the run ended right after
        this event."""
        saved = self.backend.snapshot()
        try:
            self.backend.step(valuation)
            return [
                c.name
                for c in self.backend.contracts()
                if not c.advisory and self.backend.owes(c.name)
            ]
        finally:
            self.backend.restore(saved)

    def owed_deliverables(self, agent_state: Any = None) -> list[str]:
        """The deliverables that would still be open if the agent stopped
        now: an idle event, state-only context derived, is stepped on a
        copy of the monitors."""
        if not self.deliverables:
            return []
        grounding = copy.deepcopy(self._grounding)
        grounding.current_ctx.update(self._probe_ctx(""))
        saved = self.backend.snapshot()
        try:
            self.backend.step(self._idle_valuation(agent_state, grounding))
            return [n for n in sorted(self.deliverables) if self.backend.owes(n)]
        finally:
            self.backend.restore(saved)

    @_journaled(_enc_observe)
    def observe_call(
        self, tool: str, agent_state: Any = None, *, args: dict | None = None
    ) -> list[Contract]:
        """Commit a call another agent made as an event of this trace: the
        call atom and the current facts, with no result to extract.
        ``args`` are the
        call's arguments as that agent's trace recorded them (with its
        ``succeeded`` flag), so argument predicates are judged on them."""
        event = Event(
            ts=len(self.events),
            agent=self.agent_id,
            event_type="tool_call",
            tool=tool,
            args=dict(args) if args is not None else {"succeeded": 1},
        )
        # Context derived for this call, as update() does: without it a
        # contract over ctx(...) read whatever the previous call left.
        derived = self._derived_ctx(ProposedAction(tool, dict(args or {})))
        self._grounding.current_ctx.update(derived)
        self.fact_history.append((tool, self.facts.snapshot()))
        try:
            valuation = ground_event(
                event, len(self.events), self._grounding, self.contracts.content_atoms
            )
            valuation.update(self._fact_valuation(agent_state))
            if not event.args.get("succeeded", 1):
                valuation = _attempt_only(valuation)
            violated = self.backend.step(valuation)
        finally:
            for k in derived:
                self._grounding.current_ctx.pop(k, None)
        self.events.append(event)
        return self._record_violations(tool, violated)

    def _record_violations(self, tool: str | None, violated: list[Contract]) -> list[Contract]:
        """Trace the contracts an event violated; returns the binding ones.
        An unconfirmed rule's violation is traced apart and binds nothing."""
        hard = [c for c in violated if not c.advisory]
        soft = [c.name for c in violated if c.advisory]
        if hard:
            self.tracer.record("contract/violation", tool=tool, contracts=[c.name for c in hard])
        if soft:
            self.tracer.record("rule/advisory_violated", tool=tool, contracts=soft)
        return hard

    # ------------------------------------------------------------------
    # Rules attached mid-run (acuto.rules)
    # ------------------------------------------------------------------

    def _rule_contract(self, rule_id: str) -> Contract:
        for c in self.backend.contracts():
            if c.rule == rule_id:
                return c
        raise KeyError(f"no attached rule {rule_id!r}")

    @_journaled(_enc_attach)
    def attach_rule(
        self,
        rule_id: str,
        formula: Any,
        assumption: Any = None,
        *,
        name: str,
        advisory: bool = True,
        meta: dict | None = None,
    ) -> Contract:
        """Add a rule to the library mid-run. Its monitor starts at the
        current event: the rule binds from the moment it was stated, and
        the past is not judged by it. An ``advisory`` rule (not yet
        confirmed) warns and never blocks; see :meth:`confirm_rule`.
        Formulas may be given as ``repr`` strings (a replayed journal)."""
        from contragent import parse_repr

        if rule_id in self.rules:
            raise ValueError(f"rule {rule_id!r} is already attached")
        formula = parse_repr(formula) if isinstance(formula, str) else formula
        assumption = parse_repr(assumption) if isinstance(assumption, str) else assumption
        compiled = compile_contracts(
            [CustomRule(formula, name=name, assumption=assumption, masking=not advisory)]
        )
        (contract,) = compiled.contracts
        contract = dataclasses.replace(contract, rule=rule_id, advisory=advisory, owner="user")
        self.backend.attach(contract)
        self.contracts.contracts.append(contract)
        self.contracts.fact_predicates |= compiled.fact_predicates
        # Merge per predicate: a second rule reading the same predicate
        # (two arg_field_has rules) must not drop the first one's patterns.
        for pred, atoms in compiled.content_atoms.items():
            self.contracts.content_atoms[pred] = set(
                self.contracts.content_atoms.get(pred, ())
            ) | set(atoms)
        self.rules[rule_id] = {
            "name": name,
            "status": "proposed" if advisory else "confirmed",
            "attached_at": len(self.events),
            "meta": dict(meta or {}),
        }
        self.tracer.record("rule/attached", rule=rule_id, name=name, advisory=advisory)
        return contract

    @_journaled(_enc_confirm)
    def confirm_rule(self, rule_id: str) -> None:
        """Enforce a proposed rule from now on, like an authored one. A rule
        already broken while it was advisory starts afresh."""
        old = self._rule_contract(rule_id)
        new = dataclasses.replace(old, advisory=False, masking=True)
        restart = self.backend.violated_now(old.name)
        self.backend.replace(old.name, new, restart=restart)
        self.contracts.contracts = [new if c is old else c for c in self.contracts.contracts]
        self.rules[rule_id].update(status="confirmed", confirmed_at=len(self.events))
        self.tracer.record("rule/confirmed", rule=rule_id, restarted=restart)

    @_journaled(_enc_detach)
    def detach_rule(self, rule_id: str) -> None:
        """Remove an attached rule (a revocation). The authored library
        cannot be detached: only what was attached mid-run."""
        old = self._rule_contract(rule_id)
        self.backend.detach(old.name)
        self.contracts.contracts = [c for c in self.contracts.contracts if c is not old]
        self.rules[rule_id].update(status="revoked", revoked_at=len(self.events))
        self.tracer.record("rule/revoked", rule=rule_id)

    # ------------------------------------------------------------------
    # The user changing their mind (acuto.revision)
    # ------------------------------------------------------------------

    def _active(self, name: str) -> Contract:
        for c in self.backend.contracts():
            if c.name == name:
                return c
        raise KeyError(f"no active contract {name!r}")

    @_journaled(_enc_own)
    def set_owner(self, name: str, owner: str) -> None:
        """Say who owns an authored contract: "user" when it was read off
        the user's own request, so the user may later set it aside."""
        old = self._active(name)
        new = dataclasses.replace(old, owner=owner)
        self.backend.replace(name, new)
        self.contracts.contracts = [new if c is old else c for c in self.contracts.contracts]

    @_journaled(_enc_suspend)
    def suspend(self, name: str, by: str, *, until: str | None = None, quote: str = "") -> None:
        """Set a contract aside because of something the user said (``by``
        names it: a rule or permission id). A suspended contract neither
        masks nor blocks, and is not judged; it is kept, with what
        suspended it, so :meth:`release` can bring it back. ``until=
        "request"`` suspends it for the current request only (:meth:`
        end_request`). Only contracts the user owns can be suspended."""
        self.suspenders.setdefault(by, {"until": until, "quote": quote})
        if name in self.suspended:
            self.suspended[name]["by"].add(by)
            return
        contract = self._active(name)
        if contract.owner != "user":
            raise PermissionError(f"{name!r} is not the user's to set aside")
        self.backend.detach(name)
        self.contracts.contracts = [c for c in self.contracts.contracts if c is not contract]
        self.suspended[name] = {"contract": contract, "by": {by}, "at": len(self.events)}
        if contract.rule in self.rules:
            self.rules[contract.rule]["status"] = "suspended"
        self.tracer.record("contract/suspended", contract=name, by=by, until=until)

    @_journaled(_enc_release)
    def release(self, by: str) -> list[str]:
        """Withdraw a suspender: every contract it alone kept aside is in
        force again, judged from now on (what was done while it was
        suspended is not held against it). Returns their names."""
        self.suspenders.pop(by, None)
        back = []
        for name, info in list(self.suspended.items()):
            info["by"].discard(by)
            if info["by"]:
                continue
            contract = info["contract"]
            del self.suspended[name]
            self.backend.attach(contract)
            self.contracts.contracts.append(contract)
            if contract.rule in self.rules:
                confirmed = not contract.advisory
                self.rules[contract.rule]["status"] = "confirmed" if confirmed else "proposed"
            back.append(name)
            self.tracer.record("contract/restored", contract=name, by=by)
        return back

    @_journaled(_enc_end_request)
    def end_request(self) -> list[str]:
        """The current request is over: suspensions made "for this request"
        lapse, and what they set aside is in force again."""
        back = []
        for rid, info in list(self.rules.items()):
            if info["meta"].get("until") == "request" and info["status"] in (
                "proposed",
                "confirmed",
            ):
                self.detach_rule(rid)
        for by, info in list(self.suspenders.items()):
            if info.get("until") == "request":
                back += self.release(by)
        return back

    def owed_rules(self) -> list[str]:
        """Confirmed rules that would fail if the session ended now (an
        eventuality they demand is still open)."""
        return [
            rid
            for rid, info in self.rules.items()
            if info["status"] == "confirmed" and self.backend.owes(info["name"])
        ]

    def state_digest(self) -> str:
        """A fingerprint of the control state: every monitor's residual (up
        to AC and idempotence of and/or) and assumption flag, the facts,
        the pending obligations, and the number of events. Two controllers
        with equal digests make the same decisions from here on. Context
        compaction never changes it (docs/LONG_RUNS.md, section 2)."""
        monitors = [(n, _stable(_canonical(r)), b) for n, r, b in self.backend.state_signature()]
        obligations = [(ob.desc, ob.tool, ob.triggered_at) for ob in self.obligations.pending()]
        payload = json.dumps(
            [monitors, self.facts.snapshot(), obligations, len(self.events)],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def replay(self, journal: list[dict]) -> None:
        """Feed another controller's journal into this one, in order. On a
        fresh controller this rebuilds the control state the journal
        produced: under the same contracts the same residuals, facts and
        obligations; under changed contracts the state those contracts
        would have reached on the same trace (a replayed event that
        violates a new contract is traced as a violation). The entries
        join this controller's own journal."""
        for e in journal:
            op = e["op"]
            if op == "update":
                result = ToolResult(**e["result"])
                self.update(ProposedAction(e["tool"], dict(e["args"])), result, e.get("state"))
            elif op == "assert_fact":
                self.assert_fact(
                    e["predicate"],
                    e["value"],
                    agent_state=e.get("state"),
                    checked=e.get("checked", False),
                    **e.get("metadata", {}),
                )
            elif op == "invalidate_fact":
                self.invalidate_fact(e["predicate"])
            elif op == "idle":
                self.idle(e.get("state"))
            elif op == "observe_call":
                self.observe_call(e["tool"], e.get("state"), args=e.get("args"))
            elif op == "attach_rule":
                self.attach_rule(
                    e["rule"],
                    e["formula"],
                    e.get("assumption"),
                    name=e["name"],
                    advisory=e.get("advisory", True),
                    meta=e.get("meta"),
                )
            elif op == "confirm_rule":
                self.confirm_rule(e["rule"])
            elif op == "detach_rule":
                self.detach_rule(e["rule"])
            elif op == "set_owner":
                self.set_owner(e["contract"], e["owner"])
            elif op == "suspend":
                self.suspend(e["contract"], e["by"], until=e.get("until"), quote=e.get("quote", ""))
            elif op == "release":
                self.release(e["by"])
            elif op == "end_request":
                self.end_request()
            else:
                raise ValueError(f"unknown journal entry {op!r}")

    def live_after(self, action: ProposedAction | None, agent_state: Any = None) -> bool:
        """Does the event ``action`` (an idle event when None) keep every
        guarantee satisfiable? No monitor may reach ⊥ on it, and the
        residuals it leaves must be jointly completable; an inconclusive
        completion query counts as live. Nothing is committed."""
        grounding = copy.deepcopy(self._grounding)
        if action is None:
            valuation = self._idle_valuation(agent_state, grounding)
        else:
            grounding.current_ctx.update(self._derived_ctx(action))
            valuation = self._valuation_for(action.tool, action.args, agent_state, grounding)
        if self.backend.probe(valuation, include_args_dependent=True):
            return False
        return self._completable(valuation, grounding)

    def idle_settles(self, agent_state: Any = None) -> bool:
        """Would an idle event leave every monitor in its current state?
        Then waiting longer changes nothing: with no input arriving, the
        residuals are a fixpoint of idling."""

        def signature():
            return [(n, _canonical(r), b) for n, r, b in self.backend.state_signature()]

        saved = self.backend.snapshot()
        before = signature()
        try:
            self.backend.step(self._idle_valuation(agent_state, copy.deepcopy(self._grounding)))
            return signature() == before
        finally:
            self.backend.restore(saved)

    # ------------------------------------------------------------------
    # Session end
    # ------------------------------------------------------------------

    def finalize(self) -> dict:
        """End-of-session report. Pending obligations are violations here —
        Acuto's active semantics, not the weak finite-trace collapse.

        Facts the last call's result wrote are visible only from the next
        event; when the session ends on them, the end is that event (an
        idle one), so a guarantee ``F fact(p)`` met by the last call is
        judged met, as the end gate (``owed_deliverables``) already judged
        it. The idle event is stepped on the monitors for the report only
        and not committed: the session may be continued afterwards."""
        saved = None
        if self.fact_history and self.fact_history[-1][1] != self.facts.snapshot():
            grounding = copy.deepcopy(self._grounding)
            grounding.current_ctx.update(self._probe_ctx(""))
            saved = self.backend.snapshot()
            self.backend.step(self._idle_valuation(None, grounding))
        try:
            final = self.backend.finalize()
        finally:
            if saved is not None:
                self.backend.restore(saved)
        ltlf = [r.name for r in final if not r.advisory]
        advisory = [r.name for r in final if r.advisory]
        pending = [ob.desc for ob in self.obligations.pending()]
        for desc in pending:
            self.tracer.record("obligation/pending", obligation=desc)
        for name in ltlf:
            self.tracer.record("contract/final", contract=name)
        out = {"ltlf_violations": ltlf, "pending_obligations": pending}
        if advisory:
            out["advisory_violations"] = advisory  # unconfirmed rules: reported, not binding
        statuses = self.backend.assumption_statuses()
        broken = [n for n, st in statuses.items() if st == "broken"]
        if broken:
            # The environment, not the agent: a contract whose assumption
            # was falsified is IDLE, so its guarantee no longer binds. Under
            # enforcement this list is empty — assert_fact suppresses the
            # write that would break an assumption — so a non-empty list
            # means the environment reached the control state another way.
            out["idle_contracts"] = broken
            for name in broken:
                self.tracer.record("contract/assumption_broken", contract=name)
        return out
