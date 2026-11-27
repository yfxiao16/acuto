"""Acuto (Agentic Contract-based Runtime Orchestration): a symbolic control
layer for LLM agents.

A neuro-symbolic agent harness: procedural contracts are compiled into
runtime control. The agent chooses when there is freedom; Acuto executes
when there isn't.

Three orthogonal declarations:

* **tools** — what the agent CAN do (`ToolRegistry` / `@agent.tool`)
* **flow** — how the application WANTS to organize the task
  (`FreeFlow` default, `GraphFlow` for staged tasks)
* **contracts** — what is ALLOWED and REQUIRED
  (`ContractLibrary`, attached explicitly; never encoded into the flow)

    import acuto

    agent = acuto.Agent(model=acuto.models.resolve("anthropic:claude-sonnet-5"),
                        instructions="You are a bank service agent.")

    @agent.tool
    def verify_identity(customer_id: str) -> dict:
        "Verify the customer's identity."
        ...

    library = acuto.ContractLibrary("refund-desk-v1", [
        acuto.action("issue_refund").requires("identity_verified"),
        acuto.after("verify_identity").asserts("identity_verified",
                                               when=lambda r: r.output["verified"]),
        acuto.when("fraud_detected").obligates("freeze_account", args={...}),
    ])
    library.verify()          # symbolic certificates (LTLf-SAT)
    agent.attach(library)

    result = agent.run("Refund order #123")
    for frame in agent.stream("Replace card for C-1024"):
        print(frame.kind, frame.masked)

To drive Acuto from your own loop instead, build one ``acuto.Controller``
per task and call ``route()`` / ``verify()`` / ``commit()`` (or
``execute()``) around your model and tool calls; ``acuto.Agent`` runs on
the same object.
"""

from acuto import console, models
from acuto.agent import Agent
from acuto.backend import ContractBackend, LTLfBackend
from acuto.cache import ResidualPolicyCache
from acuto.checkpoint import Checkpoint, InFlightError
from acuto.contracts import (
    ActionRule,
    AfterRule,
    CompiledContracts,
    Contract,
    CustomRule,
    WhenRule,
    action,
    after,
    ag,
    compile_contracts,
    when,
)
from acuto.control import SymbolicController
from acuto.controller import CallOutcome, Controller, as_action
from acuto.decisions import (
    Decision,
    DecisionKind,
    ProposedAction,
    ReturnVerdict,
    StepDecision,
    StepKind,
)
from acuto.facts import (
    Fact,
    FactStore,
    LocalPredicateEvaluator,
    PredicateContext,
    PredicateEvaluator,
)
from acuto.flow import Flow, FlowContext, FreeFlow, GraphFlow
from acuto.library import ContractConflictError, ContractLibrary, VerifyReport
from acuto.loop import (
    Compactor,
    ContextArchive,
    Frame,
    MaskCompactor,
    RunResult,
    SummaryCompactor,
    TruncateCompactor,
    iterate,
    run,
)
from acuto.obligations import ActiveObligation, ObligationEngine, ObligationSpec
from acuto.tools import Tool, ToolExecutor, ToolRegistry, ToolResult
from acuto.tracing import Tracer

# Back-compat alias from v0.0.1 (FreeAgent was renamed to FreeFlow).
FreeAgent = FreeFlow

__version__ = "0.1.1"

# The SDK surface, in three tiers. Everything below stays importable from
# ``acuto`` (nothing was removed); the tiers say what a user needs first.
# docs/SDK.md walks through them.
__all__ = [
    # Tier 1 — write an agent with Acuto: model, tools, contracts, flow, run.
    "Agent",
    "models",
    "console",
    "Tool",
    "ToolRegistry",
    "ContractLibrary",
    "action",
    "after",
    "when",
    "ag",
    "GraphFlow",
    "FreeFlow",
    "Frame",
    "RunResult",
    # Tier 2 — embed Acuto in your own loop: one session of control, driven
    # step by step, plus the types its methods return.
    "Controller",
    "CallOutcome",
    "as_action",
    "ProposedAction",
    "StepDecision",
    "StepKind",
    "Decision",
    "DecisionKind",
    "ReturnVerdict",
    "ToolResult",
    "run",
    "iterate",
    # Long runs: history compaction, checkpoint and resume.
    "Compactor",
    "ContextArchive",
    "MaskCompactor",
    "SummaryCompactor",
    "TruncateCompactor",
    "Checkpoint",
    "InFlightError",
    # Tier 3 — extension points and internals: custom predicates, raw
    # formulas, the residual policy cache, the underlying controller.
    "PredicateEvaluator",
    "PredicateContext",
    "LocalPredicateEvaluator",
    "Fact",
    "FactStore",
    "Flow",
    "FlowContext",
    "CustomRule",
    "Contract",
    "ActionRule",
    "AfterRule",
    "WhenRule",
    "CompiledContracts",
    "compile_contracts",
    "ContractConflictError",
    "VerifyReport",
    "SymbolicController",
    "ContractBackend",
    "LTLfBackend",
    "ResidualPolicyCache",
    "ActiveObligation",
    "ObligationEngine",
    "ObligationSpec",
    "ToolExecutor",
    "Tracer",
    "FreeAgent",
]
