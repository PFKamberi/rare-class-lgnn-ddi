"""build_sample_graph.py: class-stratified sample of the META-DDIE databases, seeded
class-0 pairs, files the other scripts read with --sample. Uses tiny databases and
random embeddings, so nothing is downloaded."""
import importlib.util
import json
import os
import pathlib
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import dgl
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PER_CLASS = 20


def _load_builder():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("build_sample_graph", SCRIPTS / "build_sample_graph.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_dbs(data_dir):
    rng = np.random.RandomState(0)
    n_drugs = 80
    ids = [f"DB{i:05d}" for i in range(1, n_drugs + 1)]
    pd.DataFrame({"id": ids, "smile": ["C" * (1 + i % 7) for i in range(n_drugs)]}).to_sql(
        "drug", sqlite3.connect(data_dir / "Drug_META_DDIE.db"), index=False)
    sizes = {**{c: 60 for c in range(1, 13)}, **{c: 8 for c in range(180, 184)},
             **{c: 4 for c in range(210, 214)}, 230: 1}
    rows = []
    for lbl, n in sizes.items():
        for _ in range(n):
            a, b = rng.choice(n_drugs, 2, replace=False)
            rows.append((ids[a], ids[b], str(lbl)))
    ev = pd.DataFrame(rows, columns=["id1", "id2", "label"])
    with sqlite3.connect(data_dir / "event_META_DDIE_5foldLabel.db") as c:   # two tables, as META-DDIE
        ev.iloc[::2].to_sql("event_a", c, index=False)
        ev.iloc[1::2].to_sql("event_b", c, index=False)
    return sizes


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    base = tmp_path_factory.mktemp("sample")
    data, work = base / "data_root", base / "work_root"
    (data / "data").mkdir(parents=True)
    work.mkdir()
    sizes = _make_dbs(data / "data")
    mod = _load_builder()
    from ddi import data as D
    from ddi.paths import get_paths
    orig = D.generate_smiles_embeddings
    gen = torch.Generator().manual_seed(0)
    D.generate_smiles_embeddings = lambda g, df, m: [torch.randn(1, 384, generator=gen) for _ in range(g.num_nodes())]
    try:
        paths = get_paths(str(data), str(work))
        paths.make_output_dirs()
        g, lg, noneg = mod.build(paths, per_class=PER_CLASS, neg_ratio=0.3, seed=0)
    finally:
        D.generate_smiles_embeddings = orig
    return SimpleNamespace(data=data, work=work, sizes=sizes, g=g, lg=lg, noneg=noneg, mod=mod, paths=paths)


def test_stratified_counts(built):
    counts = torch.bincount(built.noneg.edata["label"])
    for lbl, n in built.sizes.items():
        expected = 0 if lbl > 227 else min(n, PER_CLASS)
        got = int(counts[lbl]) if lbl < len(counts) else 0
        assert got == expected, f"class {lbl}: {got} != {expected}"


def test_negatives_are_clean(built):
    g, n_pos = built.g, built.noneg.num_edges()
    lab = g.edata["label"]
    assert (lab[:n_pos] > 0).all() and (lab[n_pos:] == 0).all()
    assert g.num_edges() - n_pos == int(n_pos * 0.3)
    u, v = g.edges()
    pos = set(zip(u[:n_pos].tolist(), v[:n_pos].tolist()))
    neg = list(zip(u[n_pos:].tolist(), v[n_pos:].tolist()))
    assert len(set(neg)) == len(neg), "duplicate class-0 pairs"
    assert all(a != b for a, b in neg), "self-loop"
    assert not any((a, b) in pos or (b, a) in pos for a, b in neg), "class-0 pair that interacts"


def test_files_and_line_graph(built):
    from ddi import graph as G
    sub, lg = G.load_graph_pair(str(built.mod.sample_pair_path(built.paths)))
    assert lg.num_nodes() == sub.num_edges()
    ref = dgl.line_graph(sub, backtracking=False)
    assert lg.num_edges() == ref.num_edges()
    noneg = dgl.load_graphs(str(built.mod.sample_drug_graph_path(built.paths)))[0][0]
    assert (noneg.edata["label"] > 0).all()


def test_seeded(built):
    ev = pd.DataFrame({"label": [1] * 50 + [2] * 3, "x": range(53)})
    a = built.mod.stratified_sample(ev, 10, 0)
    b = built.mod.stratified_sample(ev, 10, 0)
    assert a.equals(b) and (a.label.value_counts().to_dict() == {1: 10, 2: 3})


def _run(script, built, *extra):
    cmd = [sys.executable, str(SCRIPTS / script), "--data-root", str(built.data), "--work-root", str(built.work),
           "--sample", "--device", "cpu", *extra]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPTS))
    assert out.returncode == 0, f"{script} failed:\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}"
    return out.stdout


def test_scripts_run_on_the_sample(built):
    _run("02_train_warmstart.py", built, "--epochs", "1")
    assert json.loads((built.work / "warmstart" / "table_7_8_sample.json").read_text())
    _run("03_diagnostics.py", built)
    assert (built.work / "diagnostics" / "diagnostics_sample.json").exists()
    _run("04_episodic.py", built, "--epochs", "1", "--episodes", "10")
    assert (built.work / "episodic" / "episodic_results_sample.json").exists()
    out = _run("05_coldstart.py", built, "--max-epochs", "1", "--min-epochs", "1")
    assert "TABLE 8.3" in out
