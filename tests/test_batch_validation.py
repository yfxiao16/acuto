"""A batch of calls emitted in one message is validated under the formal
model's sequential semantics: call k against the state left by the admitted
calls before it, not each call against the pre-batch state.

The regression case sends seven cancellation calls in one message and
requires the controller to enforce the "at most 2" bound across the batch.
"""

from __future__ import annotations

import acuto
from acuto.decisions import DecisionKind, ProposedAction


def _controller(specs):
    return acuto.SymbolicController(specs)


def test_count_bound_holds_across_a_batch():
    c = _controller([acuto.action("cancel").at_most(2)])
    batch = [ProposedAction("cancel", {"id": str(i)}) for i in range(7)]
    kinds = [d.kind for d in c.validate_batch(batch)]
    assert kinds[:2] == [DecisionKind.ALLOW, DecisionKind.ALLOW]
    assert all(k is not DecisionKind.ALLOW for k in kinds[2:])
    # per-call validation against the pre-batch state is what let all seven through
    assert all(c.validate(a).allowed for a in batch)


def test_precedence_within_a_batch_follows_execution_order():
    c = _controller([acuto.action("b").requires_before("a")])
    a, b = ProposedAction("a", {}), ProposedAction("b", {})
    # the environment runs a batch in order, so [a, b] satisfies "a before b"
    assert [d.allowed for d in c.validate_batch([a, b])] == [True, True]
    assert [d.allowed for d in c.validate_batch([b, a])] == [False, True]


def test_batch_validation_commits_nothing():
    c = _controller([acuto.action("cancel").at_most(2)])
    batch = [ProposedAction("cancel", {}) for _ in range(3)]
    before = (c.backend.state_signature(), len(c.events), dict(c._grounding.call_counts))
    c.validate_batch(batch)
    after = (c.backend.state_signature(), len(c.events), dict(c._grounding.call_counts))
    assert before == after
    # and a real single call afterwards still sees a fresh count
    assert c.validate(batch[0]).allowed


def test_singleton_and_empty_batches_are_plain_validate():
    c = _controller([acuto.action("cancel").at_most(1)])
    one = ProposedAction("cancel", {})
    assert c.validate_batch([]) == []
    assert c.validate_batch([one])[0].kind is c.validate(one).kind


def test_blocked_sibling_does_not_advance_the_state_for_later_siblings():
    # a blocked call must not count: the third and fourth are both judged
    # against two admitted calls, so both are blocked (not just the third)
    c = _controller([acuto.action("cancel").at_most(2)])
    batch = [ProposedAction("cancel", {}) for _ in range(4)]
    allowed = [d.allowed for d in c.validate_batch(batch)]
    assert allowed == [True, True, False, False]
