"""A run that dies and is resumed: nothing done is done twice.

    python3 examples/long_run.py

An agent processes six orders, then charges the customer. With
``checkpoint_path`` Acuto writes the run's state to a file before every
step and records each call before sending it. Here the process "dies"
inside the charge. The second agent starts from the file: the six orders
are not processed again, and because the charge was sent but its outcome
is unknown, Acuto refuses to repeat it until the caller says so
(``on_in_flight="redo"``; a tool marked ``idempotent`` is simply redone).

A scripted model plays the turns, so the example needs no credentials.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
try:
    import contragent  # noqa: F401
except ModuleNotFoundError:  # development layout: sibling ContrAgent checkout
    sys.path.insert(0, str(_ROOT.parent / "ContrAgent"))

import acuto  # noqa: E402
from acuto.models import MockModel, ModelTurn, ToolCall  # noqa: E402

TURNS = [ModelTurn(tool_calls=[ToolCall("process_order", {"n": n})]) for n in range(1, 7)] + [
    ModelTurn(tool_calls=[ToolCall("charge", {"amount": 60})]),
    ModelTurn(text="Six orders processed and the customer charged once."),
]
CONTRACTS = acuto.ContractLibrary(
    "orders-v1",
    [
        acuto.action("charge").requires("orders_done").at_most(1),
        acuto.after("process_order").asserts("orders_done", when=lambda r: r.output["n"] == 6),
    ],
)


class ProcessDied(BaseException):
    """Stands in for the process being killed while a call is under way."""


def build(model, log: list[str], dies: bool) -> acuto.Agent:
    agent = acuto.Agent(model=model, instructions="Process the orders, then charge.")

    @agent.tool
    def process_order(n: int) -> dict:
        "Process one order."
        log.append(f"process_order({n})")
        return {"n": n}

    @agent.tool
    def charge(amount: int) -> dict:
        "Charge the customer."
        if dies:
            raise ProcessDied()
        log.append(f"charge({amount})")
        return {"charged": amount}

    agent.attach(CONTRACTS)
    return agent


def main() -> None:
    path = str(pathlib.Path(tempfile.mkdtemp()) / "run.json")

    first: list[str] = []
    try:
        build(MockModel(TURNS), first, dies=True).run("Process the orders.", checkpoint_path=path)
    except ProcessDied:
        print(f"first run died after {len(first)} calls: {', '.join(first)}")

    saved = acuto.Checkpoint.load(path)
    print(f"checkpoint: step {saved.step}, call in flight: {saved.in_flight[0]['tool']}")

    second: list[str] = []
    agent = build(MockModel(TURNS[saved.step :]), second, dies=False)
    try:
        agent.run("Process the orders.", resume=acuto.Checkpoint.load(path))
    except acuto.InFlightError as exc:
        print(f"resume refused: {exc}")
    result = agent.run(
        "Process the orders.", resume=acuto.Checkpoint.load(path), on_in_flight="redo"
    )
    print(f"resumed run made {len(second)} call: {', '.join(second)}")
    print(f"status {result.status}; violations {result.finalize['ltlf_violations']}")
    assert first == [f"process_order({n})" for n in range(1, 7)] and second == ["charge(60)"]


if __name__ == "__main__":
    main()
