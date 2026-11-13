"""The scripts in scripts/ run end to end in test mode on a synthetic thesis-like graph
(CPU, a few minutes). The synthetic graph is written where 01_prepare_data.py expects the
extracted thesis .bin, so nothing is downloaded."""
import json
import os
import pathlib
import subprocess
import sys

import dgl
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from synthetic import make_synthetic_drug_graph  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
N = 120


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    base = tmp_path_factory.mktemp("scripts")
    data, work = base / "data", base / "work"
    data.mkdir()
    work.mkdir()
    drug = make_synthetic_drug_graph()
    dgl.save_graphs(str(work / "ddi_graph_and_linegraph.bin"),
                    [drug, dgl.line_graph(drug, backtracking=False)])
    return data, work


def _run(script, roots, *extra):
    data, work = roots
    cmd = [sys.executable, str(SCRIPTS / script), "--data-root", str(data), "--work-root", str(work),
           "--test", "--n-test-drugs", str(N), "--device", "cpu", *extra]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPTS))
    assert out.returncode == 0, f"{script} failed:\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}"
    return out.stdout


def test_01_prepare_data(roots):
    out = _run("01_prepare_data.py", roots)
    _, work = roots
    assert (work / "test_graphs" / f"test_pair_{N}.bin").exists()
    assert (work / "test_graphs" / f"test_drug_graph_noneg_{N}.bin").exists()
    assert "test line graph" in out


def test_02_train_warmstart(roots):
    _run("02_train_warmstart.py", roots)
    _, work = roots
    t = json.loads((work / "warmstart" / f"table_7_8_test{N}.json").read_text())
    assert any("baseline" in k for k in t)


def test_03_diagnostics(roots):
    _run("03_diagnostics.py", roots)
    _, work = roots
    d = json.loads((work / "diagnostics" / f"diagnostics_test{N}.json").read_text())
    assert {"table_7_10", "table_7_9", "table_7_11", "knn_purity"} <= set(d)


def test_04_episodic(roots):
    _run("04_episodic.py", roots, "--episodes", "20")
    _, work = roots
    for f in (f"episodic_results_test{N}.json", f"t3_chemberta_only_test{N}.json", f"t4_smoothing_test{N}.json"):
        assert (work / "episodic" / f).exists()


def test_05_coldstart(roots):
    out = _run("05_coldstart.py", roots)
    assert "TABLE 8.3" in out


def test_missing_roots_fail_clearly(tmp_path):
    out = subprocess.run([sys.executable, str(SCRIPTS / "01_prepare_data.py")], capture_output=True,
                         text=True, cwd=str(SCRIPTS), env={k: v for k, v in os.environ.items()
                                                           if not k.startswith("DDI_")})
    assert out.returncode != 0 and "--data-root" in (out.stdout + out.stderr)
