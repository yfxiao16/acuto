"""The package reaches ContrAgent through its public modules only."""

import ast
import pathlib

PUBLIC = {"contragent", "contragent.formulas", "contragent.kernel"}


def test_acuto_imports_only_public_contragent_modules():
    root = pathlib.Path(__file__).resolve().parent.parent / "acuto"
    found = set()
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("contragent"):
                found.add(node.module)
            elif isinstance(node, ast.Import):
                found |= {a.name for a in node.names if a.name.startswith("contragent")}
    assert found and found <= PUBLIC, sorted(found - PUBLIC)
