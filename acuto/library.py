"""ContractLibrary: the named, verifiable collection of contracts.

The library is the static declaration of the RIGHT rail (control state):
it compiles once into `CompiledContracts` and is attached to an Agent.
`verify()` produces symbolic certificates via ContrAgent's LTLf
satisfiability engine — no LLM, no network:

* per-contract **satisfiable** (the contract is not vacuously false),
* per-contract **falsifiable** (the contract is not a tautology — it can actually
  fire),
* **jointly satisfiable** (the library is conflict-free: some session keeps
  every assumption and meets every guarantee at once).

`False` answers from the SAT engine are sound; `None` means the search
budget was exceeded (reported as ``unknown``, never an error).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

from contragent.formulas import Implies, Not
from contragent.kernel import is_satisfiable

from acuto.contracts import CompiledContracts, Contract, compile_contracts


class ContractConflictError(Exception):
    """Raised by verify() when a certificate definitively fails."""


@dataclass
class ContractReport:
    name: str
    satisfiable: bool | None
    falsifiable: bool | None
    # Compatibility: does a valid environment exist? A `False` here means no
    # run can keep the assumption, so the contract can never be in force.
    assumption_satisfiable: bool | None = None

    @property
    def ok(self) -> bool:
        return (
            self.satisfiable is not False
            and self.falsifiable is not False
            and self.assumption_satisfiable is not False
        )


@dataclass
class VerifyReport:
    library: str
    contracts: list[ContractReport] = field(default_factory=list)
    jointly_satisfiable: bool | None = None

    @property
    def ok(self) -> bool:
        return self.jointly_satisfiable is not False and all(r.ok for r in self.contracts)

    @property
    def unknown(self) -> list[str]:
        out = [r.name for r in self.contracts if r.satisfiable is None or r.falsifiable is None]
        if self.jointly_satisfiable is None:
            out.append("<joint>")
        return out

    def render(self) -> str:
        lines = [f"ContractLibrary {self.library!r}: {'OK' if self.ok else 'CONFLICT'}"]
        for r in self.contracts:
            mark = "ok" if r.ok else "FAIL"
            lines.append(f"  [{mark}] {r.name} (sat={r.satisfiable}, falsifiable={r.falsifiable})")
        lines.append(f"  joint satisfiability: {self.jointly_satisfiable}")
        return "\n".join(lines)


class ContractLibrary:
    """Named collection of contract specs; compile once, reuse across runs.

    The assume-guarantee data architecture mirrors ContrAgent's
    ``Contract`` model: one assumption over environment predicates, one
    guarantee over the agent's own behaviour, lists meaning AND.
    ``from_contragent`` imports its libraries directly."""

    def __init__(self, name: str, specs: Iterable) -> None:
        self.name = name
        self.specs = list(specs)
        self._compiled: CompiledContracts | None = None
        self.verified: VerifyReport | None = None  # the last verify() report

    @classmethod
    def from_contragent(
        cls, source, *, agent: str = "*", name: str | None = None, soft=None
    ) -> "ContractLibrary":
        """Import a ContrAgent contract library.

        ``source`` is either an iterable of ``contragent.models.contract.
        Contract`` objects or a path to a ContrAgent YAML config. A/G structure
        is preserved: each guarantee becomes one Contract whose assumption is
        the conjunction of the source contract's assumptions, and Acuto
        reads that assumption exactly as ContrAgent does — an obligation on
        the environment, checked at compile time to be a safety property
        over environment predicates. Non-deterministic (sto) constraints
        are skipped.

        ``soft`` is an optional predicate over the source contract object:
        where it returns True the imported contract joins the soft tier
        (``masking=False``) — enforced at validate with feedback, but never
        silently removing tools from the exposed set. Use it for trace-mined
        convention patterns, whose precision does not earn masking.
        """
        from functools import reduce

        from acuto.contracts import CustomRule

        def _raw(f):
            return getattr(f, "formula", f)

        specs: list = []

        def _add(assumptions, guarantees, desc, masking=True):
            raw_as = [_raw(a) for a in assumptions]
            assumption = reduce(lambda x, y: x & y, raw_as) if raw_as else None
            for i, g in enumerate(guarantees):
                suffix = f" [{i}]" if len(guarantees) > 1 else ""
                specs.append(
                    CustomRule(
                        formula=_raw(g),
                        assumption=assumption,
                        name=(desc or "imported") + suffix,
                        masking=masking,
                    )
                )

        if isinstance(source, (str, __import__("pathlib").Path)):
            from contragent import load_config
            from contragent.kernel import resolve_entry

            config = load_config(str(source))
            ag_cfg = config.agents.get(agent) or next(iter(config.agents.values()))
            for ce in ag_cfg.contracts:
                desc = getattr(ce, "desc", None) or ""

                def _parts(part):
                    if part is None:
                        return []
                    entries = part if isinstance(part, list) else [part]
                    out = []
                    for e in entries:
                        _nl, parsed = resolve_entry(e)
                        if parsed is not None and getattr(parsed, "is_det", True):
                            out.append(parsed)
                    return out

                _add(_parts(ce.assumption), _parts(ce.guarantee), desc)
        else:
            for contract in source:
                if hasattr(contract, "is_pure_det") and not contract.is_pure_det:
                    continue  # sto contracts are out of the deterministic layer
                _add(
                    list(getattr(contract, "assumptions", []) or []),
                    list(getattr(contract, "guarantees", []) or []),
                    getattr(contract, "desc", None),
                    masking=not (soft is not None and soft(contract)),
                )
        return cls(name or f"contragent:{agent}", specs)

    @property
    def compiled(self) -> CompiledContracts:
        if self._compiled is None:
            self._compiled = compile_contracts(self.specs)
        return self._compiled

    @property
    def contracts(self) -> list[Contract]:
        return self.compiled.contracts

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self) -> Iterator:
        return iter(self.specs)

    def to_contragent(self, *, agent_id: str = "acuto"):
        """Reverse adapter: this library as ContrAgent ``Contract`` objects
        (for ContrAgent's analyses, e.g. the MUC conflict check)."""
        from contragent import Agent as CAgent
        from contragent import Contract as CContract

        owner = CAgent(id=agent_id, tools=[])
        return [
            CContract(
                agent=owner,
                guarantee=c.formula,
                assumption=c.assumption,
                desc=c.name,
            )
            for c in self.contracts
        ]

    def check_conflicts(self, **kwargs):
        """Deep library-level conflict check via ContrAgent: one joint
        satisfiability query over ⋀(Aᵢ∧Gᵢ), with a minimal unsatisfiable
        core extracted to locate the contracts to repair when it fails.
        Returns ContrAgent's ConflictReport (`.ok`, `.render()`).
        Complements `verify()`, which certifies per-contract
        satisfiability and joint satisfiability."""
        from contragent import check_conflicts

        return check_conflicts(self.to_contragent(), **kwargs)

    def auditor(self, *, predicate_evaluator=None, agent_id: str = "auditor"):
        """A passive observe-mode controller over this library: it sees every
        committed action of a run (any condition), records violations, and
        never filters or blocks — including on the environment side, where an
        enforcing controller would suppress a fact write that breaks an
        assumption and the auditor instead records the break. Attach via
        ``agent.run(..., auditor=...)`` /
        the loop's ``auditor=`` parameter — this is how procedural-compliance
        metrics stay condition-independent (enforce runs should audit clean;
        baseline runs reveal their violations)."""
        from acuto.control import SymbolicController
        from acuto.tracing import Tracer

        return SymbolicController(
            self.compiled,
            predicate_evaluator=predicate_evaluator,
            agent_id=agent_id,
            tracer=Tracer(),
            enforcing=False,
        )

    def verify(self, *, max_states: int = 20_000, strict: bool = True) -> VerifyReport:
        """Symbolic verification. With ``strict=True`` (default), a
        definitive failure raises :class:`ContractConflictError`."""
        report = VerifyReport(library=self.name)
        # Per contract, the two classical certificates: consistency, i.e. a
        # valid implementation exists (A -> G satisfiable), and falsifiability,
        # i.e. the guarantee can actually be broken while the environment keeps
        # its side (A /\ ~G satisfiable). Compatibility, a valid environment
        # exists, is the assumption's own satisfiability.
        folds = [
            Implies(contract.assumption, contract.formula)
            if contract.assumption is not None
            else contract.formula
            for contract in self.contracts
        ]
        for contract, fold in zip(self.contracts, folds):
            sat = is_satisfiable([fold], max_states=max_states)
            fal = is_satisfiable([Not(fold)], max_states=max_states)
            a_sat = (
                is_satisfiable([contract.assumption], max_states=max_states)
                if contract.assumption is not None
                else None
            )
            report.contracts.append(
                ContractReport(
                    name=contract.name,
                    satisfiable=sat,
                    falsifiable=fal,
                    assumption_satisfiable=a_sat,
                )
            )
        # The library query is ContrAgent's: is /\_i (A_i /\ G_i) satisfiable?
        # An assumption holds on every legal run, so the joint model must keep
        # every assumption and meet every guarantee at once; folding each pair
        # into A_i -> G_i would instead certify libraries that are conflict-free
        # only because some assumption fails.
        conjuncts: list = []
        for contract in self.contracts:
            conjuncts.append(contract.formula)
            if contract.assumption is not None:
                conjuncts.append(contract.assumption)
        if conjuncts:
            report.jointly_satisfiable = is_satisfiable(conjuncts, max_states=max_states)
        else:
            report.jointly_satisfiable = True
        self.verified = report
        if strict and not report.ok:
            raise ContractConflictError(report.render())
        return report
