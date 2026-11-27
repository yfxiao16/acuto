"""Dev convenience: if contragent is not installed (``pip install -e
".[dev]"`` installs it from PyPI), fall back to a sibling checkout
(../ContrAgent)."""

import pathlib
import sys

try:
    import contragent  # noqa: F401
except ModuleNotFoundError:
    sibling = pathlib.Path(__file__).resolve().parent.parent / "ContrAgent"
    if (sibling / "contragent").exists():
        sys.path.insert(0, str(sibling))
