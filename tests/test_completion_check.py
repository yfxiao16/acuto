"""The completion_check mask: hiding a call that dooms the session.

The one-step mask asks each contract on its own, so a sequence of
individually legal calls can reach a state no continuation completes.
With ``completion_check=True`` the controller also asks the joint question.
"""

import acuto
from acuto.control import SymbolicController
from contragent.formulas.parser import parse_repr


def _library():
    """``a`` obliges ``b`` next, and ``b`` may be called at most once."""
    return acuto.ContractLibrary(
        "dead-end-v1",
        [
            acuto.CustomRule(
                formula=parse_repr("G(called('a') -> X(called('b')))"),
                name="a obliges b next",
            ),
            acuto.CustomRule(
                formula=parse_repr("G((Var('count','b') <= 1))"),
                name="b at most once",
            ),
        ],
    )


def _controller(**kwargs):
    return SymbolicController(_library(), **kwargs)


def _commit(controller, tool):
    """Commit one successful call of ``tool`` into the controller."""
    action = acuto.ProposedAction(tool, {})
    controller.update(action, acuto.ToolResult(tool=tool, args={}, ok=True))


def test_one_step_mask_admits_the_dead_end_call():
    """Baseline: with the bound spent, ``a`` still looks admissible."""
    controller = _controller()
    _commit(controller, "b")
    assert "a" in controller.admissible_actions(["a", "b"])


def test_completion_check_mask_hides_it():
    controller = _controller(completion_check=True)
    _commit(controller, "b")
    assert "a" not in controller.admissible_actions(["a", "b"])


def test_completion_check_mask_leaves_a_healthy_state_alone():
    """Before the bound is spent, ``a`` is admissible either way."""
    plain = _controller()
    guarded = _controller(completion_check=True)
    assert "a" in plain.admissible_actions(["a", "b"])
    assert "a" in guarded.admissible_actions(["a", "b"])


def test_flag_is_off_by_default():
    assert _controller().completion_check is False


def test_validation_blocks_the_dooming_call():
    """The same joint question, asked with the arguments in hand."""
    controller = _controller(completion_check=True)
    _commit(controller, "b")
    decision = controller.validate(acuto.ProposedAction("a", {}))
    assert not decision.allowed


def test_validation_allows_it_before_the_bound_is_spent():
    controller = _controller(completion_check=True)
    assert controller.validate(acuto.ProposedAction("a", {})).allowed


def test_validation_unaffected_when_the_flag_is_off():
    controller = _controller()
    _commit(controller, "b")
    assert controller.validate(acuto.ProposedAction("a", {})).allowed
