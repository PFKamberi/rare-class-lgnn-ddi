"""
Warm-start reporting evaluation (Table 7.8).

Source: rerun_hidden384.ipynb cell 2 (ch7_rerun_FINAL.py: strict metrics on
the canonical split) and cell 5 (five evaluation draws per checkpoint, mean and
population sd). Those are the numbers in Table 7.8.

Every draw re-samples neighbours ([15, 10], batch 1024, unseeded, as in the
thesis), so draws differ; the spread is the sampling noise reported as +/- sd.
"""

import os
import json

import numpy as np
import torch
from dgl.dataloading import DataLoader, NeighborSampler

from .metrics import strict_metrics
from .model import GraphSAGEBlock, load_checkpoint
from .retrieval import build_retrieval_x_base
from .train import collect_preds_labels

# (tag, checkpoint file, fusion_type, proto_mode, k) -- rerun_hidden384 cell 2
THESIS_RUNS = [
    ("baseline (no retrieval)",     "ckpt_baseline.pt",                   "set", None,     None),
    ("retrieval: additive, k=3",    "ckpt_proto_alpha_rare_only_k3.pt",   "set", "alpha",  3),
    ("retrieval: additive, k=5",    "ckpt_proto_alpha_rare_only_k5.pt",   "set", "alpha",  5),
    ("retrieval: gated, k=5",       "ckpt_proto_gate_rare_only_k5.pt",    "set", "gate",   5),
    ("retrieval: concat, k=5 (full model)", "ckpt_proto_concat_rare_only_k5.pt", "set", "concat", 5),
]
# The thesis took the baseline row from a single draw stored in
# thesis_control_preds.npz; here it gets five draws like the rest.


def tail_from_split(g, num_classes, rare_thr=15):
    train_mask = g.ndata["train_mask"].bool()
    train_counts = torch.bincount(g.ndata["label"][train_mask], minlength=num_classes)
    return (train_counts < rare_thr).numpy(), train_counts


def build_model_for_checkpoint(g, num_classes, fusion, pmode, k, hidden=None, rare_thr=15,
                               device="cuda", knn_cache_path=None):
    in_feats = g.ndata["drug_u_emb"].shape[1]
    hidden = hidden or num_classes
    pm, rm = (None, None)
    if pmode is not None:
        pm, rm = build_retrieval_x_base(g, num_classes, proto_k=k, proto_tau=0.07,
                                        proto_weighting="softmax", rare_threshold=rare_thr,
                                        device=device, knn_cache_path=knn_cache_path)
    return GraphSAGEBlock(
        in_feats=in_feats, hidden_feats=hidden, num_classes=num_classes,
        dropout=0.1, fusion_type=fusion, fusion_heads=4,
        proto_module=pm, proto_fusion_mode=(pmode or "alpha"),
        proto_apply=("rare_only" if pmode else "none"),
        rare_node_mask=rm).to(device)


def evaluate_draws(model, g, test_nids, num_classes, tail, n_draws=5, device="cuda",
                   fanouts=(15, 10), batch_size=1024):
    """n_draws inference passes; returns list of strict-metric dicts and the
    last draw's (y, p)."""
    rows = []
    y = p = None
    for _ in range(n_draws):
        ld = DataLoader(g, test_nids, NeighborSampler(list(fanouts)),
                        batch_size=batch_size, shuffle=False, num_workers=0)
        y, p = collect_preds_labels(model, ld, device)
        y, p = np.asarray(y), np.asarray(p)
        rows.append(strict_metrics(y, p, num_classes, tail))
    return rows, (y, p)


def summarise(rows):
    """mean and population sd (np.std, ddof=0) as in rerun_hidden384 cell 5."""
    out = {}
    for key in ("acc", "evt", "rare", "rec"):
        a = np.array([r[key] for r in rows], dtype=float)
        out[key] = (float(a.mean()), float(a.std()))
    out["n_rare"] = rows[0]["n"]
    return out


def evaluate_checkpoint_dir(g, ckpt_dir, runs=THESIS_RUNS, n_draws=5, rare_thr=15,
                            device="cuda", knn_cache_dir=None, out_json=None, hidden_for=None):
    """Evaluate every checkpoint in `runs` that exists in ckpt_dir."""
    num_classes = int(g.ndata["label"].max().item() + 1)
    tail, _ = tail_from_split(g, num_classes, rare_thr)
    test_nids = g.ndata["test_mask"].nonzero(as_tuple=False).squeeze(1)
    hidden_for = hidden_for or {}
    results = {}
    for tag, fname, fusion, pmode, k in runs:
        path = os.path.join(ckpt_dir, fname)
        if not os.path.exists(path):
            print(f"SKIP {tag}: {path} not found")
            continue
        cache = (os.path.join(knn_cache_dir, f"knn_cache_k{k}.pt")
                 if (knn_cache_dir and k) else None)
        model = build_model_for_checkpoint(g, num_classes, fusion, pmode, k,
                                           hidden=hidden_for.get(fname), rare_thr=rare_thr,
                                           device=device, knn_cache_path=cache)
        load_checkpoint(model, path, device=device)
        model.eval()
        rows, _ = evaluate_draws(model, g, test_nids, num_classes, tail, n_draws, device)
        s = summarise(rows)
        results[tag] = {"draws": rows, "summary": s}
        print(f"{tag:<40} acc {s['acc'][0]:.4f}  macroF1 {s['evt'][0]:.4f}+/-{s['evt'][1]:.4f} | "
              f"rareF1 {s['rare'][0]:.4f}+/-{s['rare'][1]:.4f} | rec {s['rec'][0]:.1f}/{s['n_rare']}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if out_json:
            with open(out_json, "w") as f:
                json.dump(results, f, indent=2)
    n_rare_cls = int(tail.sum())
    n_rare_present = int(len(np.intersect1d(np.arange(num_classes)[tail],
                                            g.ndata["label"][test_nids].numpy())))
    print(f"\nrare-F1 ceiling = {n_rare_present}/{n_rare_cls} = "
          f"{n_rare_present / max(n_rare_cls, 1):.4f} (rare classes absent from test score 0)")
    return results
