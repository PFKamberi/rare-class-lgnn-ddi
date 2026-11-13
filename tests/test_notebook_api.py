import ast
import importlib
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").rglob("*.ipynb"))
if not NOTEBOOKS:
    pytest.skip("no notebooks in this checkout (they are not part of the repository)",
                allow_module_level=True)


def _code(nb_path):
    cells = json.loads(nb_path.read_text())["cells"]
    for c in cells:
        if c["cell_type"] != "code":
            continue
        lines = [l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("!", "%"))]
        yield "\n".join(lines)


def _uses(nb_path):
    """(module, attribute) pairs used by the notebook."""
    aliases, uses = {}, []
    for src in _code(nb_path):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "ddi":
                for a in node.names:
                    full = f"{node.module}.{a.name}"
                    try:
                        importlib.import_module(full)          # a submodule: record its alias
                        aliases[a.asname or a.name] = full
                    except ImportError:
                        uses.append((node.module, a.name))     # a function / constant
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] == "ddi":
                        aliases[a.asname or a.name] = a.name
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id in aliases):
                uses.append((aliases[node.value.id], node.attr))
    return uses


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_uses_existing_ddi_names(nb_path):
    missing = []
    for mod, attr in _uses(nb_path):
        if not hasattr(importlib.import_module(mod), attr):
            missing.append(f"{mod}.{attr}")
    assert not missing, f"{nb_path.name} uses names that do not exist: {sorted(set(missing))}"


def test_parser_sees_known_uses():
    nb = ROOT / "notebooks" / "01_warmstart_retrieval.ipynb"
    if not nb.exists():
        pytest.skip("notebook 01 not present")
    used = {f"{m}.{a}" for m, a in _uses(nb)}
    assert "ddi.splits.filter_and_split_graph" in used
    assert "ddi.graph.attach_endpoint_features" in used
