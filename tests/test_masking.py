"""Dynamic action-space compilation: admissible_actions over facts,
precedence, rate limits, and invalidation."""

from __future__ import annotations

import acuto
from tests.conftest import execute_and_commit


def test_requires_masks_until_facts_established(controller, registry):
    candidates = registry.names()

    admissible = controller.admissible_actions(candidates)
    assert "issue_refund" not in admissible
    assert "verify_identity" in admissible
    assert "check_fraud" in admissible

    execute_and_commit(controller, registry, "verify_identity", {"user_id": "u1"})
    assert "issue_refund" not in controller.admissible_actions(candidates)

    execute_and_commit(controller, registry, "check_fraud", {"user_id": "u1"})
    assert "issue_refund" in controller.admissible_actions(candidates)


def test_invalidation_retracts_admissibility(controller, registry):
    candidates = registry.names()
    execute_and_commit(controller, registry, "verify_identity", {"user_id": "u1"})
    execute_and_commit(controller, registry, "check_fraud", {"user_id": "u1"})
    assert "issue_refund" in controller.admissible_actions(candidates)

    execute_and_commit(
        controller, registry, "change_identity_document", {"user_id": "u1", "document": "passport"}
    )
    assert not controller.facts.truthy("identity_verified")
    assert "issue_refund" not in controller.admissible_actions(candidates)


def test_rate_limit_masks_after_use(controller, registry):
    candidates = registry.names()
    execute_and_commit(controller, registry, "verify_identity", {"user_id": "u1"})
    execute_and_commit(controller, registry, "check_fraud", {"user_id": "u1"})
    execute_and_commit(controller, registry, "issue_refund", {"order_id": "o1", "amount": 10.0})
    # at_most(1): a second refund is no longer admissible
    assert "issue_refund" not in controller.admissible_actions(candidates)


def test_forbidden_when_is_hard_block(controller, registry, world):
    world["fraud"] = True
    execute_and_commit(controller, registry, "verify_identity", {"user_id": "u1"})
    execute_and_commit(controller, registry, "check_fraud", {"user_id": "u1"})
    assert controller.facts.truthy("fraud_detected")

    decision = controller.validate(
        acuto.ProposedAction("issue_refund", {"order_id": "o1", "amount": 5.0})
    )
    assert decision.kind is acuto.DecisionKind.BLOCK


def test_validate_returns_require_with_hints(controller):
    decision = controller.validate(
        acuto.ProposedAction("issue_refund", {"order_id": "o1", "amount": 5.0})
    )
    assert decision.kind is acuto.DecisionKind.REQUIRE
    assert "identity_verified" in decision.requirements
    assert any("verify_identity" in h for h in decision.hints)
    # The feedback message names the concrete recovery tool.
    assert "verify_identity" in decision.message(acuto.ProposedAction("issue_refund", {}))


def test_precedence_rule():
    reg = acuto.ToolRegistry()

    @reg.tool
    def check_policy() -> dict:
        "Check policy."
        return {}

    @reg.tool
    def approve_loan(loan_id: str) -> dict:
        "Approve a loan."
        return {}

    controller = acuto.SymbolicController(
        [acuto.action("approve_loan").requires_before("check_policy")], registry=reg
    )
    assert "approve_loan" not in controller.admissible_actions(reg.names())
    execute_and_commit(controller, reg, "check_policy", {})
    assert "approve_loan" in controller.admissible_actions(reg.names())


def _derived_contracts():
    return [
        acuto.action("issue_refund").requires("ready"),
        acuto.after("verify_identity").asserts("identity_verified"),
        acuto.after("check_fraud").asserts("fraud_checked"),
    ]


class _Derived(acuto.LocalPredicateEvaluator):
    BASE = ("identity_verified", "fraud_checked")

    def evaluate(self, predicate, context):
        if predicate == "ready":
            return all(context.facts.get(f) is not None for f in self.BASE)
        return super().evaluate(predicate, context)


def test_require_hint_never_names_a_derived_predicate_as_a_tool(registry):
    """A predicate the evaluator derives is asserted by no tool. Without a
    ``missing`` hook the hint must still say it is a condition, never leave
    the bare name for the model to propose as a tool call."""
    c = acuto.SymbolicController(
        _derived_contracts(), registry=registry, predicate_evaluator=_Derived()
    )
    refund = acuto.ProposedAction("issue_refund", {"order_id": "o1", "amount": 5.0})
    d = c.validate(refund)
    assert d.kind is acuto.DecisionKind.REQUIRE
    assert d.requirements == ["ready"]
    msg = d.message(refund)
    assert "call `ready`" not in msg
    assert "not a tool" in msg


def test_require_hint_expands_derived_predicate_to_missing_tools(registry):
    class WithMissing(_Derived):
        def missing(self, predicate, context):
            if predicate == "ready":
                return [f for f in self.BASE if context.facts.get(f) is None]
            return None

    c = acuto.SymbolicController(
        _derived_contracts(), registry=registry, predicate_evaluator=WithMissing()
    )
    refund = acuto.ProposedAction("issue_refund", {"order_id": "o1", "amount": 5.0})
    d = c.validate(refund)
    assert d.kind is acuto.DecisionKind.REQUIRE
    assert any("verify_identity" in h for h in d.hints)
    assert any("check_fraud" in h for h in d.hints)
    assert "call `ready`" not in d.message(refund)

    execute_and_commit(c, registry, "verify_identity", {"user_id": "u1"})
    d = c.validate(refund)
    assert d.kind is acuto.DecisionKind.REQUIRE
    assert not any("verify_identity" in h for h in d.hints)
    assert any("check_fraud" in h for h in d.hints)

    execute_and_commit(c, registry, "check_fraud", {"user_id": "u1"})
    assert c.validate(refund).kind is acuto.DecisionKind.ALLOW
