"""Contract backend: assume-guarantee monitoring via LTLf residuals.

Each contract gets a guarantee monitor and, when it carries an assumption,
an assumption monitor. Runtime semantics, following ContrAgent's
three-valued contract valuation (IDLE / ACTIVE / FAIL):

* the guarantee is in force from the start of the session — the agent is
  bound by it while the environment keeps its side of the contract;
* an assumption is what the environment is required to keep. The contract
  falls IDLE the moment an environment event falsifies it (residual ⊥),
  and an IDLE contract constrains the agent no further;
* Acuto does not wait for that to happen. Like ContrAgent it maintains an
  assumption by *suppressing* the environment event that would falsify it
  (:meth:`acuto.control.SymbolicController.assert_fact`), so under
  enforcement every loaded contract stays ACTIVE and IDLE is a diagnostic
  of an environment Acuto could not hold to its side.

The split form is what makes that possible: folding the pair into the
single formula ``A → G`` would let a broken assumption silently discharge
the guarantee instead of surfacing as an environment violation. It also
exposes per-side diagnostics, drives A/G-aware feedback, and matches
ContrAgent's ``Contract`` model when importing its libraries. The rest of
Acuto depends only on the :class:`ContractBackend` protocol — probe /
step / finalize over valuation dicts.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contragent.kernel import DFAEvaluator, make_evaluator

from acuto.contracts import Contract

Valuation = dict[str, object]


@runtime_checkable
class ContractBackend(Protocol):
    def probe(
        self,
        valuation: Valuation,
        *,
        include_args_dependent: bool = True,
        masking_only: bool = False,
    ) -> list[Contract]: ...

    def step(self, valuation: Valuation) -> list[Contract]: ...

    def finalize(self) -> list[Contract]: ...


class _Monitor:
    __slots__ = ("contract", "g", "a", "a_broken")

    def __init__(self, contract: Contract) -> None:
        self.contract = contract
        self.g = make_evaluator(contract.formula)
        self.a = DFAEvaluator(contract.assumption) if contract.assumption is not None else None
        self.a_broken = False  # the environment has falsified its side

    def _advance_assumption(self, valuation: Valuation) -> None:
        if self.a is None or self.a_broken:
            return
        if self.a.step(valuation) == "⊥":
            self.a_broken = True

    def step(self, valuation: Valuation) -> bool:
        """Advance one event; True iff the contract is (now) violated.

        The guarantee monitor advances either way, so a contract that goes
        IDLE keeps a well-defined residual for the control-state
        signature; only the verdict is withheld."""
        self._advance_assumption(valuation)
        g_verdict = self.g.step(valuation)
        return not self.a_broken and g_verdict == "⊥"

    def assumption_broken_by(self, valuation: Valuation) -> bool:
        """What-if on the environment side: would this event falsify the
        assumption? Leaves the monitor untouched. ``False`` for an
        unconditional contract and for one already IDLE."""
        if self.a is None or self.a_broken:
            return False
        snap = self.a.snapshot()
        verdict = self.a.step(valuation)
        self.a.restore(snap)
        return verdict == "⊥"

    def snapshot(self) -> tuple:
        """Restorable copy of this monitor's state (both DFAs + the flag)."""
        return (
            self.g.snapshot(),
            self.a.snapshot() if self.a is not None else None,
            self.a_broken,
        )

    def restore(self, snap: tuple) -> None:
        g_snap, a_snap, broken = snap
        self.g.restore(g_snap)
        if a_snap is not None:
            self.a.restore(a_snap)
        self.a_broken = broken

    def probe(self, valuation: Valuation) -> bool:
        snap = self.snapshot()
        violated = self.step(valuation)
        self.restore(snap)
        return violated

    def final_violated(self) -> bool:
        if self.a_broken:
            return False
        return self.g.finalize() == "⊥"

    @property
    def assumption_status(self) -> str:
        if self.contract.assumption is None:
            return "unconditional"
        return "broken" if self.a_broken else "held"


class LTLfBackend:
    """One monitor pair per contract; violations are contracts whose guarantee is ⊥
    while their assumption still holds.

    Because Acuto blocks violating actions *before* they commit, the live
    monitors only ever consume compliant events in enforce flow, so
    probing needs no staleness bookkeeping.
    """

    def __init__(self, contracts: list[Contract]) -> None:
        self._monitors = [_Monitor(contract) for contract in contracts]

    def probe(
        self,
        valuation: Valuation,
        *,
        include_args_dependent: bool = True,
        masking_only: bool = False,
        include_advisory: bool = False,
    ) -> list[Contract]:
        """What-if: would this event violate any contract? Leaves state untouched.

        ``masking_only`` restricts the probe to contracts that participate in
        tool masking (``Contract.masking``); soft-tier contracts still fire
        at validate, where the violation reaches the model as feedback.
        Unconfirmed (advisory) rules are left out unless ``include_advisory``:
        only validate asks for them, to warn."""
        return [
            m.contract
            for m in self._monitors
            if (include_args_dependent or not m.contract.args_dependent)
            and (not masking_only or m.contract.masking)
            and (include_advisory or not m.contract.advisory)
            and m.probe(valuation)
        ]

    def step(self, valuation: Valuation) -> list[Contract]:
        """Commit one event into every monitor; returns contracts now violated."""
        return [m.contract for m in self._monitors if m.step(valuation)]

    def finalize(self) -> list[Contract]:
        """Session-end verdicts (weak finite-trace collapse)."""
        return [m.contract for m in self._monitors if m.final_violated()]

    def snapshot(self) -> list:
        """Restorable copy of every monitor's state; pairs with :meth:`restore`
        so a caller can step the monitors speculatively (a batch of calls the
        model emitted together) and put them back."""
        return [m.snapshot() for m in self._monitors]

    def restore(self, snaps: list) -> None:
        for m, snap in zip(self._monitors, snaps):
            m.restore(snap)

    def residuals_after(
        self,
        valuation: Valuation,
        *,
        include_args_dependent: bool = False,
        masking_only: bool = True,
    ) -> list:
        """Guarantee residuals the monitors would hold after ``valuation``.

        Speculative like :meth:`probe`: the monitors are stepped, read, and
        restored. The residuals are what a co-accessibility query needs --
        what each contract still demands of the continuation. Contracts
        whose environment side has already broken are dropped, since their
        guarantee no longer binds.
        """
        snaps = self.snapshot()
        try:
            out = []
            for m in self._monitors:
                if m.a_broken:
                    continue
                if not include_args_dependent and m.contract.args_dependent:
                    continue
                if masking_only and not m.contract.masking:
                    continue
                if m.contract.advisory:  # an unconfirmed rule constrains nothing
                    continue
                m.step(valuation)
                out.append(m.g.residual)
            return out
        finally:
            self.restore(snaps)

    def state_signature(self) -> tuple:
        """Hashable snapshot of the symbolic control state (per-contract
        residuals + the environment-side flag). Together with the fact snapshot and
        counters this keys the future residual-policy cache (partial
        evaluation of contracts per control state — the symbolic-layer
        analogue of prefix/KV reuse); tonight it instruments how often
        control states REPEAT across tasks (the potential hit rate)."""
        return tuple((m.contract.name, m.g.residual, m.a_broken) for m in self._monitors)

    def assumption_breakers(self, valuation: Valuation) -> list[Contract]:
        """Contracts whose assumption this environment event would falsify.

        The environment-side counterpart of :meth:`probe`: the controller
        suppresses such an event rather than let it discharge a guarantee
        (ContrAgent's bidirectional enforcement)."""
        return [
            m.contract
            for m in self._monitors
            if not m.contract.advisory and m.assumption_broken_by(valuation)
        ]

    def assumption_statuses(self) -> dict[str, str]:
        """contract name -> unconditional | held | broken."""
        return {m.contract.name: m.assumption_status for m in self._monitors}

    def contracts(self) -> list[Contract]:
        return [m.contract for m in self._monitors]

    def attach(self, contract: Contract) -> None:
        """Monitor one more contract, from the current event on."""
        if any(m.contract.name == contract.name for m in self._monitors):
            raise ValueError(f"a contract named {contract.name!r} is already monitored")
        self._monitors.append(_Monitor(contract))

    def detach(self, name: str) -> Contract:
        """Stop monitoring a contract; returns it."""
        for i, m in enumerate(self._monitors):
            if m.contract.name == name:
                return self._monitors.pop(i).contract
        raise KeyError(name)

    def replace(self, name: str, contract: Contract, *, restart: bool = False) -> None:
        """Swap a monitored contract's record (say, advisory to enforced),
        keeping its monitor state, or starting it afresh with ``restart``."""
        for i, m in enumerate(self._monitors):
            if m.contract.name == name:
                if restart:
                    self._monitors[i] = _Monitor(contract)
                else:
                    m.contract = contract
                return
        raise KeyError(name)

    def violated_now(self, name: str) -> bool:
        """Has this contract's guarantee already reached ⊥?"""
        for m in self._monitors:
            if m.contract.name == name:
                return m.g.peek() == "⊥" and not m.a_broken
        raise KeyError(name)

    def owes(self, name: str) -> bool:
        """Would this contract fail if the session ended now?"""
        for m in self._monitors:
            if m.contract.name == name:
                return m.final_violated()
        raise KeyError(name)
