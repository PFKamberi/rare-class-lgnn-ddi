"""End-to-end smoke tests on a synthetic graph (CPU, about a minute).

    pytest -q tests
"""
import os
import sys

import numpy as np
import pytest
import torch
import dgl
from dgl.dataloading import DataLoader, NeighborSampler
from dgl.nn import SAGEConv

sys.path.insert(0, os.path.dirname(__file__))
from synthetic import make_synthetic_drug_graph  # noqa: E402

from ddi import graph as G
from ddi import splits, train, evaluate
from ddi.experiments import diagnostics, episodic, coldstart
from ddi.model import ScatterSAGEConv, GraphSAGEBlock

DEV = torch.device("cpu")


@pytest.mark.parametrize("fin,fout", [(16, 8), (8, 16)])
def test_scatter_sage_matches_dgl(fin, fout):
    g = dgl.rand_graph(200, 2000)
    lg = dgl.line_graph(g, backtracking=False)
    lg.ndata["x"] = torch.randn(lg.num_nodes(), fin)
    _, _, blocks = next(iter(DataLoader(lg, torch.arange(50), NeighborSampler([4]), batch_size=50)))
    x = blocks[0].srcdata["x"]
    ref = SAGEConv(fin, fout, aggregator_type="mean")
    ours = ScatterSAGEConv(fin, fout)
    ours.load_state_dict(ref.state_dict())
    with torch.no_grad():
        assert torch.allclose(ref(blocks[0], x), ours(blocks[0], x), atol=1e-5)


@pytest.fixture(scope="module")
def warm_graph(tmp_path_factory):
    d = tmp_path_factory.mktemp("warm")
    drug = make_synthetic_drug_graph()
    sub, lg = G.make_test_subgraph(drug, n_drugs=120, seed=0)
    path = str(d / "pair.bin")
    dgl.save_graphs(path, [sub, lg])
    sub, lg = G.load_graph_pair(path)
    lg = G.attach_endpoint_features(sub, lg)
    return splits.filter_and_split_graph(lg), d


def test_warmstart_train_and_eval(warm_graph):
    g, d = warm_graph
    ck = str(d / "ckpt")
    for name in ("baseline", "proto_concat_rare_only_k5"):
        kw = dict(train.WARM_START_VARIANTS[name])
        kw.update(num_epochs=2, batch_size=256, dropout=0.1, fusion_type="set", fusion_heads=4)
        train.run_variant(g, name, checkpoint_dir=ck, knn_cache_path=str(d / "knn_k5.pt"),
                          device=DEV, **kw)
    res = evaluate.evaluate_checkpoint_dir(g, ck, n_draws=2, device=DEV, knn_cache_dir=str(d))
    assert "baseline (no retrieval)" in res


def test_knn_cache_shape_guard(warm_graph, tmp_path):
    g, _ = warm_graph
    bad = str(tmp_path / "bad.pt")
    torch.save({"knn_idx": torch.zeros(3, 5, dtype=torch.long), "knn_sim": torch.zeros(3, 5)}, bad)
    from ddi.retrieval import build_retrieval_x_base
    with pytest.raises(ValueError):
        build_retrieval_x_base(g, 214, device=DEV, knn_cache_path=bad)


def test_diagnostics(warm_graph):
    g, d = warm_graph
    ck = str(d / "ckpt")
    ctrl = diagnostics.control_logits(g, os.path.join(ck, "ckpt_baseline.pt"),
                                      os.path.join(ck, "ckpt_proto_concat_rare_only_k5.pt"),
                                      DEV, knn_cache_path=str(d / "knn_k5.pt"))
    C = int(g.ndata["label"].max()) + 1
    tc, tail = diagnostics.train_counts_and_tail(g, C)
    diagnostics.control_table(ctrl, tail, C)
    diagnostics.frequency_groups(ctrl, C)
    cache = torch.load(str(d / "knn_k5.pt"))
    diagnostics.label_dependence(ctrl, g, cache["knn_idx"], cache["knn_sim"], tc, n_boot=50)
    diagnostics.knn_purity(g, cache["knn_idx"], cache["knn_sim"])


def test_episodic(tmp_path):
    drug = make_synthetic_drug_graph(seed=1)
    sub, lg = G.make_test_subgraph(drug, n_drugs=150, seed=0)
    lg = G.attach_endpoint_features(sub, lg)
    g = splits.build_metaddie_graph(lg)
    ck = str(tmp_path / "pre.pt")
    model = episodic.train_common_only(g, DEV, ck, epochs=2, patience=5)
    Z = episodic.compute_all_embeddings(model, g, DEV)
    labels_dev, pools = episodic.make_pools(g, DEV)
    res = episodic.run_episodic_suite(Z, labels_dev, pools, episodes=20, seed=42, ways=(2, 3))
    assert res
    Zc = episodic.build_chemberta_only_embeddings(g, DEV, "set_untrained")
    assert Zc.shape[0] == g.num_nodes()
    tr = g.ndata["train_mask"].nonzero(as_tuple=False).squeeze(1)
    ki, ks = episodic.build_knn_to_train(Z, tr, k=5)
    mask = (labels_dev >= 176)
    Zs = episodic.compute_smoothed_Z(Z, ki, ks, mask, "alpha", fusion_params={"alpha": 0.7})
    assert Zs.shape == Z.shape


def test_coldstart(tmp_path):
    drug = make_synthetic_drug_graph(n_drugs=200, n_edges=9000, seed=2, with_negatives=False)
    d = coldstart.ColdStartDirs(str(tmp_path / "ch8"), tag="test")
    coldstart.phase0_from_drug_graph(d, drug)
    coldstart.phase0_report(d)
    coldstart.phase1(d)
    small = dict(max_epochs=2, min_epochs=1)
    coldstart.phase2(d, DEV, runs=coldstart.E1_RUNS[2:4], **small)
    coldstart.phase3(d, DEV, runs=coldstart.E2_RUNS[3:], **small)
    coldstart.phase4(d, DEV)
    coldstart.fix_e1_4(d, DEV)
    coldstart.fix_e2_4(d, DEV)
    r2, r3, r4 = coldstart.phase5_tables(d)
    assert "E1_4_set_ldam_proto (fixed index)" in r2
    fixed = r2["E1_4_set_ldam_proto (fixed index)"]
    assert fixed["S2_setupA"]["oob_lookups"] == 0
