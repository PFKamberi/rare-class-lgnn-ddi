import pathlib

import numpy as np

from ddi.experiments import diagnostics
from ddi.metrics import strict_metrics

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "ddi"


def test_no_212_metric_names_left():
    text = "\n".join(p.read_text() for p in SRC.rglob("*.py"))
    for name in ("all212", "macro212"):
        assert name not in text, f"{name} still in src/ddi"


def test_strict_metrics_keys():
    y = np.array([0, 1, 1, 2, 3])
    p = np.array([0, 1, 2, 2, 3])
    tail = np.array([False, False, False, True])
    m = strict_metrics(y, p, 4, tail)
    assert "all212" not in m and "evt" in m and "rare" in m


def test_control_table_columns():
    rng = np.random.default_rng(0)
    C, n = 5, 40
    y = rng.integers(0, C, n)
    z = rng.normal(size=(n, C))
    ctrl = dict(y_true=y, logits_base=z, y_true_s=y, logits_smooth=z)
    tail = np.array([False, False, False, True, True])
    rows = diagnostics.control_table(ctrl, tail, C)
    assert len(rows) == 3
    for r in rows:
        assert set(r) == {"model", "tailF1", "macroEvt", "rare_correct", "n_rare"}
