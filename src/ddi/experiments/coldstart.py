"""
Cold-start evaluation (Chapter 8).

Source: the LOCAL ch8_rebuild_resumable.ipynb (RTX 5090 version, the one that
produced Tables 8.1-8.7), cells 2-24, including the retrieval-index fix in
cells 22-23 that the thesis E1.4 / E2.4 rows come from.

Every phase writes to disk and skips work that is already there, as in the
notebook. Paths come from a ColdStartDirs object instead of module globals.

KNOWN ISSUES (kept):
  * the thesis parent graph was built with list(set(...)) node ids, whose order
    depended on Python's string-hash seed. phase0_build_parent now sorts the ids,
    so rebuilds are deterministic, but they give a DIFFERENT drug split from the
    thesis. Reproduce the thesis only from the cached parent_graph.bin
    (graphs_v4 on the laptop).
  * retrieval routing (rare_node_mask) uses the true label, as in warm start.
  * train-node rows of the union index are recomputed with a different
    self-exclusion rule than at training time (see retrieval.build_union_index).
"""

import gc
import json
import os
import pickle
import sqlite3
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import dgl
from dgl.dataloading import DataLoader, NeighborSampler

from ..losses import FocalLoss, LDAMLoss, class_weights
from ..metrics import full_metrics
from ..model import GraphSAGEBlock, load_checkpoint
from ..data import chunk_and_embed
from ..retrieval import SimilarityPrototype, build_topk_cosine_knn, build_union_index

# notebook constants (cell 2)
SEED, FANOUTS, BATCH_SIZE, DROPOUT = 42, [15, 10], 1024, 0.1
MAX_EPOCHS, PATIENCE, MIN_DELTA, MIN_EPOCHS = 100, 20, 1e-3, 30
TAIL_THRESHOLDS = (15, 20)
SLICES = ["S2_setupA", "S2_setupB", "S3_setupA", "S3_setupB"]
SLICE_GROUPS = {"S2": ["S2_setupA", "S2_setupB"], "S3": ["S3_setupA", "S3_setupB"]}
FILTER_RULE = "total>=4"   # matches Ch4 4.5.1, keeps 211 classes. NOT "train>=4" (176).
RARE_THRESHOLD, PROTO_K, PROTO_TAU = 15, 5, 0.07
BUCKET_NAMES = {0: "TRAIN", 1: "VAL", 2: "S2", 3: "S3", -1: "DISCARD"}

E1_RUNS = [
    ("E1_1_gated_ce",       dict(fusion_type="gated", use_proto=False), "cross_entropy"),
    ("E1_2_gated_ldam",     dict(fusion_type="gated", use_proto=False), "ldam"),
    ("E1_3_set_ldam",       dict(fusion_type="set",   use_proto=False), "ldam"),
    ("E1_4_set_ldam_proto", dict(fusion_type="set",   use_proto=True, proto_mode="alpha"), "ldam"),
    ("E1_5_concat_ldam",    dict(fusion_type="concat", use_proto=False), "ldam"),
]
E2_RUNS = [
    ("E2_1_gated_ce",       dict(fusion_type="gated", use_proto=False), "cross_entropy"),
    ("E2_2_gated_ldam",     dict(fusion_type="gated", use_proto=False), "ldam"),
    ("E2_3_set_ldam",       dict(fusion_type="set",   use_proto=False), "ldam"),
    ("E2_4_set_ldam_proto", dict(fusion_type="set",   use_proto=True, proto_mode="alpha"), "ldam"),
]


@dataclass
class ColdStartDirs:
    """Same layout as the thesis run: <base>/graphs_<tag>, checkpoints_<tag>, results_<tag>."""
    base: str
    tag: str = "v4"

    def __post_init__(self):
        self.gdir = os.path.join(self.base, f"graphs_{self.tag}")
        self.ckpt = os.path.join(self.base, f"checkpoints_{self.tag}")
        self.out = os.path.join(self.base, f"results_{self.tag}")
        self.pred = os.path.join(self.out, "preds")
        self.pred_fixed = os.path.join(self.out, "preds_fixed")
        for d in (self.base, self.gdir, self.ckpt, self.out, self.pred, self.pred_fixed):
            os.makedirs(d, exist_ok=True)
        self.parent = os.path.join(self.gdir, "parent_graph.bin")
        self.parent_r = os.path.join(self.gdir, "parent_relabelled.bin")
        self.meta = os.path.join(self.gdir, "filter_meta.pkl")
        self.ctl_meta = os.path.join(self.gdir, "control_meta.pkl")
        self.manifest = os.path.join(self.out, "manifest.json")


def banner(t):
    print("\n" + "=" * 76 + f"\n{t}\n" + "=" * 76, flush=True)


def mem(tag=""):
    try:
        import psutil
        print(f"    [mem] {tag} {psutil.Process().memory_info().rss/1e9:.1f} GB", flush=True)
    except Exception:
        pass


def load_manifest(d):
    return json.load(open(d.manifest)) if os.path.exists(d.manifest) else {}


def mark(d, key, val=True):
    m = load_manifest(d)
    m[key] = val
    json.dump(m, open(d.manifest, "w"), indent=2)

def build_lg(parent, eids):
    sub = dgl.edge_subgraph(parent, eids, relabel_nodes=True, store_ids=True)
    lg = dgl.line_graph(sub, backtracking=False, shared=False)
    u, v = sub.edges()
    f = sub.ndata["smiles_embedding"]
    lg.ndata["drug_u_emb"] = f[u]
    lg.ndata["drug_v_emb"] = f[v]
    lg.ndata["label"] = sub.edata["label"]
    lg.ndata["parent_eid"] = sub.edata[dgl.EID]
    return lg


def build_setup(parent, train_eids, test_eids, strict):
    """Union line graph of train + held-out interactions. Setup A (strict=True)
    removes every held-out<->held-out edge; Setup B keeps them."""
    lg = build_lg(parent, torch.cat([train_eids, test_eids]))
    nt, ne = train_eids.numel(), test_eids.numel()
    is_test = torch.zeros(lg.num_nodes(), dtype=torch.bool)
    is_test[nt:nt + ne] = True
    if strict:
        src, dst = lg.edges()
        rm = (is_test[src] & is_test[dst]).nonzero(as_tuple=False).squeeze(-1)
        if rm.numel():
            lg = dgl.remove_edges(lg, rm)
    return lg, is_test.nonzero(as_tuple=False).squeeze(-1)


def gpath(d, name):
    return os.path.join(d.gdir, f"lg_{name}.bin")


def save_graph(d, name, g, test_nodes=None):
    dgl.save_graphs(gpath(d, name), [g])
    if test_nodes is not None:
        torch.save(test_nodes, os.path.join(d.gdir, f"testnodes_{name}.pt"))


def load_graph(d, name):
    g = dgl.load_graphs(gpath(d, name))[0][0]
    tp = os.path.join(d.gdir, f"testnodes_{name}.pt")
    return g, (torch.load(tp) if os.path.exists(tp) else None)


def graph_exists(d, name):
    return os.path.exists(gpath(d, name))


# ---------------------------------------------------------------------------
# phase 0: parent graph and drug-disjoint split
# ---------------------------------------------------------------------------
def assign_drug_buckets(g, seed=SEED):
    """70/15/15 drug split; edge buckets 0=train-train, 1=val-val,
    2=S2 (one train drug, one test drug), 3=S3 (both test), -1=discarded."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(g.num_nodes())
    n = g.num_nodes()
    ntr, nva = int(round(.70 * n)), int(round(.15 * n))
    da = torch.full((n,), -1, dtype=torch.long)
    da[perm[:ntr]] = 0
    da[perm[ntr:ntr + nva]] = 1
    da[perm[ntr + nva:]] = 2
    u, v = g.edges()
    au, av = da[u], da[v]
    eb = torch.full((g.num_edges(),), -1, dtype=torch.long)
    eb[(au == 0) & (av == 0)] = 0
    eb[(au == 1) & (av == 1)] = 1
    eb[((au == 0) & (av == 2)) | ((au == 2) & (av == 0))] = 2
    eb[(au == 2) & (av == 2)] = 3
    g.ndata["drug_bucket"] = da
    g.edata["edge_bucket"] = eb
    return g


def phase0_build_parent(d, drug_db, event_db, seed=SEED):
    """Build the no-negative parent drug graph from the raw databases."""
    from transformers import AutoTokenizer, AutoModel

    def ldf(db, t):
        c = sqlite3.connect(db)
        df = pd.read_sql_query(f"SELECT * FROM {t}", c)
        c.close()
        return df

    c = sqlite3.connect(event_db)
    cur = c.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tabs = cur.fetchall()
    c.close()
    ev = pd.concat([ldf(event_db, t[0]) for t in tabs])
    ev["label"] = ev["label"].astype(int)
    ev = ev[ev.label <= 227]
    drugs = ldf(drug_db, "drug")
    ids = sorted(set(ev["id1"]) | set(ev["id2"]))   # deterministic: sorted DrugBank ids
    i2n = {v: i for i, v in enumerate(ids)}
    g = dgl.graph(([i2n[x] for x in ev["id1"]], [i2n[x] for x in ev["id2"]]), num_nodes=len(ids))
    g.edata["label"] = torch.tensor(ev.label.to_list(), dtype=torch.long)
    tok = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    mdl = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM").eval()
    inv = {v: k for k, v in i2n.items()}
    sm = [drugs[drugs.id == inv[i]]["smile"].values[0] for i in range(g.num_nodes())]

    g.ndata["smiles_embedding"] = torch.stack(
        [chunk_and_embed(s, tok, mdl) for s in tqdm(sm, desc="ChemBERTa")], 0).squeeze(1)
    g = assign_drug_buckets(g, seed)
    dgl.save_graphs(d.parent, [g])
    print("  saved", d.parent)
    return g


def phase0_from_drug_graph(d, drug_graph, seed=SEED):
    """TEST ONLY: use an existing drug graph (e.g. the test subgraph built with
    drop_negatives=True from the thesis .bin) instead of rebuilding from the
    databases. Same bucket rule."""
    g = dgl.graph(drug_graph.edges(), num_nodes=drug_graph.num_nodes())
    g.ndata["smiles_embedding"] = drug_graph.ndata["smiles_embedding"]
    g.edata["label"] = drug_graph.edata["label"]
    g = assign_drug_buckets(g, seed)
    dgl.save_graphs(d.parent, [g])
    print("  saved", d.parent)
    return g


def phase0_report(d):
    _g = dgl.load_graphs(d.parent)[0][0]
    da, eb = _g.ndata["drug_bucket"], _g.edata["edge_bucket"]
    print(f"  drugs train/val/test: {(da==0).sum()}/{(da==1).sum()}/{(da==2).sum()}")
    for b in (0, 1, 2, 3, -1):
        n_ = int((eb == b).sum())
        print(f"    {BUCKET_NAMES[b]:<8}{n_:>10,}  {100*n_/_g.num_edges():5.2f}%")
    S = [set((da == i).nonzero().flatten().tolist()) for i in range(3)]
    print(f"  intersections (all 0): {len(S[0]&S[1])},{len(S[0]&S[2])},{len(S[1]&S[2])}")


# ---------------------------------------------------------------------------
# phase 1: class filter and line graphs
# ---------------------------------------------------------------------------
JOBS = [("train", None, None), ("val", None, None),
        ("S2_setupA", "S2", True), ("S2_setupB", "S2", False),
        ("S3_setupA", "S3", True), ("S3_setupB", "S3", False)]


def phase1(d, filter_rule=FILTER_RULE):
    if os.path.exists(d.meta):
        meta = pickle.load(open(d.meta, "rb"))
        print(f"  cached: rule={meta['rule']}  C={meta['C']}")
    else:
        g = dgl.load_graphs(d.parent)[0][0]
        lab, eb = g.edata["label"], g.edata["edge_bucket"]
        Craw = int(lab.max()) + 1
        ct = torch.bincount(lab, minlength=Craw)
        c_tr = torch.bincount(lab[eb == 0], minlength=Craw)
        rules = {"total>=4": ct >= 4, "total>=2": ct >= 2, "train>=4": c_tr >= 4}
        keep = rules[filter_rule] & (ct > 0)
        kept = keep.nonzero().flatten()
        remap = torch.full((Craw,), -1, dtype=torch.long)
        remap[kept] = torch.arange(kept.numel())
        C = int(kept.numel())
        g.edata["label"] = remap[lab].clamp(min=0)
        ke = keep[lab]
        print(f"  rule={filter_rule}  C={C}")
        print("  POST-FILTER buckets (Table 8.1):")
        tot = int(ke.sum())
        post = {}
        for b in (0, 1, 2, 3, -1):
            n_ = int(((eb == b) & ke).sum())
            post[BUCKET_NAMES[b]] = n_
            print(f"    {BUCKET_NAMES[b]:<8}{n_:>10,}  {100*n_/tot:5.2f}%")
        eids = {k: ((eb == b) & ke).nonzero(as_tuple=False).squeeze(-1)
                for b, k in [(0, "train"), (1, "val"), (2, "S2"), (3, "S3")]}
        train_counts = torch.bincount(g.edata["label"][eids["train"]], minlength=C)
        meta = {"C": C, "train_counts": train_counts, "eids": eids,
                "rule": filter_rule, "post_filter_buckets": post}
        pickle.dump(meta, open(d.meta, "wb"))
        dgl.save_graphs(d.parent_r, [g])
        del g
        gc.collect()

    eids = meta["eids"]
    for name, bucket, strict in JOBS:
        if graph_exists(d, name):
            print(f"  {name}: cached")
            continue
        print(f"  building {name} ...", flush=True)
        t0 = time.time()
        parent = dgl.load_graphs(d.parent_r)[0][0]
        if bucket is None:
            gg = build_lg(parent, eids[name])
            tn = None
        else:
            gg, tn = build_setup(parent, eids["train"], eids[bucket], strict)
        save_graph(d, name, gg, tn)
        print(f"    nodes {gg.num_nodes():,} edges {gg.num_edges():,} ({time.time()-t0:.0f}s)")
        del gg, parent, tn
        gc.collect()
        mem(name)

    print("\n  TABLE 8.2:")
    for name, _, _ in JOBS:
        gg, _ = load_graph(d, name)
        print(f"    {name:<11} nodes {gg.num_nodes():>10,} edges {gg.num_edges():>14,}")
        del gg
        gc.collect()
    return meta


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_preds(model, loader):
    model.eval()
    P, Y = [], []
    for _, _, blocks in loader:
        Y.append(blocks[-1].dstdata["label"].cpu())
        P.append(model(blocks).argmax(1).cpu())
    return (torch.cat(Y).numpy(), torch.cat(P).numpy()) if P else (None, None)


def train_model(d, model, g_train, g_val, opt, sch, run, device, loss_type="ldam", label_smoothing=None,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, min_epochs=MIN_EPOCHS):
    """Resumable: saves _last.pt every epoch and _best.pt on improvement of
    STRICT validation macro-F1 (labels=arange(C))."""
    last_p = os.path.join(d.ckpt, f"ckpt_{run}_last.pt")
    best_p = os.path.join(d.ckpt, f"ckpt_{run}_best.pt")
    C_ = model.classifier.out_features
    lt = g_train.ndata["label"]
    cc = torch.bincount(lt, minlength=C_).float()
    cw = class_weights(lt, C_, device)
    ldam = LDAMLoss(cc.tolist()).to(device) if loss_type == "ldam" else None
    samp = NeighborSampler(FANOUTS)
    vl = DataLoader(g_val, torch.arange(g_val.num_nodes()), samp, batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=0)
    start, best, best_ep, noimp = 1, -1e9, 0, 0
    if os.path.exists(last_p):
        st = torch.load(last_p, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if sch and "sch" in st:
            sch.load_state_dict(st["sch"])
        start = st["epoch"] + 1
        best, best_ep, noimp = st["best"], st["best_ep"], st["noimp"]
        print(f"  RESUMED {run} at epoch {start} (best {best:.4f}@{best_ep})", flush=True)
    for ep in range(start, max_epochs + 1):
        dl = DataLoader(g_train, torch.arange(g_train.num_nodes()), samp,
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        model.train()
        tl, nb = 0.0, 0
        for _, _, blocks in tqdm(dl, desc=f"{run} ep{ep:03d}", leave=False):
            y = blocks[-1].dstdata["label"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(blocks)
            if loss_type == "ldam":
                loss = ldam(logits, y)
            elif loss_type == "focal":
                loss = FocalLoss(alpha=cw)(logits, y)
            else:
                loss = F.cross_entropy(logits, y, weight=cw, label_smoothing=label_smoothing or 0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            nb += 1
        if sch:
            sch.step()
        yv, pv = collect_preds(model, vl)
        per = f1_score(yv, pv, labels=np.arange(C_), average=None, zero_division=0)
        vs = float(per.mean())
        vd = float(f1_score(yv, pv, average="macro", zero_division=0))
        imp = vs > best + MIN_DELTA
        if imp:
            best, best_ep, noimp = vs, ep, 0
            torch.save({"model": model.state_dict(), "best": best, "best_ep": ep,
                        "val_default": vd}, best_p)
        else:
            noimp += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sch": sch.state_dict() if sch else None, "epoch": ep, "best": best,
                    "best_ep": best_ep, "noimp": noimp}, last_p)
        print(f"  ep{ep:03d} loss={tl/max(nb,1):.4f} val_strict={vs:.4f} val_default={vd:.4f} "
              f"best={best:.4f}@{best_ep} noimp={noimp}/{patience}{' *' if imp else ''}", flush=True)
        if ep >= min_epochs and noimp >= patience:
            print(f"  early stop @{ep}")
            break
    st = torch.load(best_p, map_location=device)
    model.load_state_dict(st["model"])
    return {"best_val_strict": st["best"], "best_epoch": st["best_ep"],
            "best_val_default": st["val_default"]}


def make_model(d, fusion_type, C_, in_feats, g_train, device, use_proto=False, proto_mode="alpha",
               proto_k=PROTO_K, rare_thr=RARE_THRESHOLD):
    """Hidden width = C (211), as in the thesis run."""
    proto, rare = None, None
    if use_proto:
        X = (0.5 * (g_train.ndata["drug_u_emb"] + g_train.ndata["drug_v_emb"])).detach().cpu().float()
        tl = g_train.ndata["label"].cpu()
        rare = (torch.bincount(tl, minlength=C_) < rare_thr)[tl]
        # cache is graph-specific; keyed by node count so train and ctl_train never share
        cp = os.path.join(d.ckpt, f"knn_cache_k{proto_k}_n{g_train.num_nodes()}.pt")
        if os.path.exists(cp):
            c = torch.load(cp, map_location="cpu")
            ki, ks = c["knn_idx"], c["knn_sim"]
        else:
            ki, ks = build_topk_cosine_knn(X, torch.arange(g_train.num_nodes()), k=proto_k, device=device)
            torch.save({"knn_idx": ki, "knn_sim": ks}, cp)
        proto = SimilarityPrototype(X, ki, ks)
    return GraphSAGEBlock(in_feats, C_, C_, dropout=DROPOUT, fusion_type=fusion_type,
                          proto_module=proto, proto_fusion_mode=proto_mode,
                          proto_apply="rare_only" if use_proto else "all",
                          rare_node_mask=rare).to(device)


@torch.no_grad()
def eval_slice(d, model, name, run, tcounts, Cn, pred_dir=None):
    g, tn = load_graph(d, name)
    if tn is None:
        tn = torch.arange(g.num_nodes())
    dl = DataLoader(g, tn, NeighborSampler(FANOUTS), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    y, p = collect_preds(model, dl)
    np.savez_compressed(os.path.join(pred_dir or d.pred, f"preds_{run}_{name}.npz"), y_true=y, y_pred=p)
    m = full_metrics(y, p, Cn, tcounts, TAIL_THRESHOLDS)
    print(f"    {name:<11} acc={m['acc']:.4f} strict={m['macro_f1_strict']:.4f} "
          f"(default {m['macro_f1_sklearn_default']:.4f}) tail15={m['tail_f1_thr15']:.4f}", flush=True)
    del g
    gc.collect()
    return m


def _new_opt(model, max_epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)
    return opt, sch


# ---------------------------------------------------------------------------
# phase 2: cold-start runs
# ---------------------------------------------------------------------------
def phase2(d, device, runs=E1_RUNS, max_epochs=MAX_EPOCHS, min_epochs=MIN_EPOCHS):
    RP = os.path.join(d.out, "results_phase2.json")
    res = json.load(open(RP)) if os.path.exists(RP) else {}
    meta = pickle.load(open(d.meta, "rb"))
    C, train_counts = meta["C"], meta["train_counts"]
    gtr, _ = load_graph(d, "train")
    gva, _ = load_graph(d, "val")
    in_feats = gtr.ndata["drug_u_emb"].shape[1]
    for run, kw, loss in runs:
        if run in res:
            print(f"  {run}: done")
            continue
        print(f"\n  >>> {run} ({loss})", flush=True)
        torch.manual_seed(SEED)
        model = make_model(d, C_=C, in_feats=in_feats, g_train=gtr, device=device, **kw)
        opt, sch = _new_opt(model, max_epochs)
        tr = train_model(d, model, gtr, gva, opt, sch, run, device, loss_type=loss,
                         label_smoothing=0.05 if loss == "cross_entropy" else None,
                         max_epochs=max_epochs, min_epochs=min_epochs)
        row = dict(tr)
        for sl in SLICES:
            row[sl] = eval_slice(d, model, sl, run, train_counts, C)
        if getattr(model, "_oob", 0):
            row["oob_lookups"] = model._oob
        res[run] = row
        json.dump(res, open(RP, "w"), indent=2)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del gtr, gva
    gc.collect()
    mark(d, "phase2")
    print("\n  saved", RP)
    return res


# ---------------------------------------------------------------------------
# phase 3: warm-start control inside the training bucket
# ---------------------------------------------------------------------------
def phase3(d, device, runs=E2_RUNS, max_epochs=MAX_EPOCHS, min_epochs=MIN_EPOCHS):
    meta = pickle.load(open(d.meta, "rb"))
    C = meta["C"]
    if os.path.exists(d.ctl_meta):
        ctl_counts = pickle.load(open(d.ctl_meta, "rb"))["train_counts"]
    else:
        gtr, _ = load_graph(d, "train")
        pe = gtr.ndata["parent_eid"]
        y = gtr.ndata["label"].numpy()
        idx = np.arange(len(y))
        ok = np.isin(y, np.where(np.bincount(y, minlength=C) >= 4)[0])
        idx = idx[ok]
        tr_i, tmp = train_test_split(idx, test_size=(1 - 0.70), random_state=SEED, stratify=y[idx])
        ok2 = np.isin(y[tmp], np.where(np.bincount(y[tmp], minlength=C) >= 2)[0])
        tmp = tmp[ok2]
        va_i, te_i = train_test_split(tmp, test_size=0.5, random_state=SEED, stratify=y[tmp])
        print(f"  control split train/val/test: {len(tr_i):,}/{len(va_i):,}/{len(te_i):,}")
        parent = dgl.load_graphs(d.parent_r)[0][0]
        for nm, ii in [("ctl_train", tr_i), ("ctl_val", va_i)]:
            if not graph_exists(d, nm):
                gg = build_lg(parent, pe[ii])
                save_graph(d, nm, gg)
                del gg
                gc.collect()
        if not graph_exists(d, "ctl_test"):
            gg, tn = build_setup(parent, pe[tr_i], pe[te_i], strict=False)
            save_graph(d, "ctl_test", gg, tn)
            del gg, tn
            gc.collect()
        gg, _ = load_graph(d, "ctl_train")
        ctl_counts = torch.bincount(gg.ndata["label"], minlength=C)
        del gg
        pickle.dump({"train_counts": ctl_counts}, open(d.ctl_meta, "wb"))
        del parent, gtr
        gc.collect()

    RP3 = os.path.join(d.out, "results_phase3.json")
    res3 = json.load(open(RP3)) if os.path.exists(RP3) else {}
    gtr, _ = load_graph(d, "ctl_train")
    gva, _ = load_graph(d, "ctl_val")
    in_feats = gtr.ndata["drug_u_emb"].shape[1]
    for run, kw, loss in runs:
        if run in res3:
            print(f"  {run}: done")
            continue
        print(f"\n  >>> {run} ({loss})", flush=True)
        torch.manual_seed(SEED)
        model = make_model(d, C_=C, in_feats=in_feats, g_train=gtr, device=device, **kw)
        opt, sch = _new_opt(model, max_epochs)
        tr = train_model(d, model, gtr, gva, opt, sch, run, device, loss_type=loss,
                         label_smoothing=0.05 if loss == "cross_entropy" else None,
                         max_epochs=max_epochs, min_epochs=min_epochs)
        row = dict(tr)
        row["test"] = eval_slice(d, model, "ctl_test", run, ctl_counts, C)
        res3[run] = row
        json.dump(res3, open(RP3, "w"), indent=2)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del gtr, gva
    gc.collect()
    mark(d, "phase3")
    print("\n  saved", RP3)
    return res3


# ---------------------------------------------------------------------------
# phase 4: frozen ChemBERTa nearest-prototype floor
# ---------------------------------------------------------------------------
def phase4(d, device):
    meta = pickle.load(open(d.meta, "rb"))
    C, train_counts = meta["C"], meta["train_counts"]
    gtr, _ = load_graph(d, "train")
    feats = 0.5 * (gtr.ndata["drug_u_emb"] + gtr.ndata["drug_v_emb"])
    tl = gtr.ndata["label"]
    P = torch.zeros(C, feats.shape[1])
    pres = torch.zeros(C, dtype=torch.bool)
    for c_ in range(C):
        m = tl == c_
        if m.any():
            P[c_] = feats[m].mean(0)
            pres[c_] = True
    del gtr, feats
    gc.collect()
    Pn = F.normalize(P, dim=-1).to(device)
    Pn[~pres.to(device)] = -1e4
    res4 = {}
    for sl in SLICES:
        g, tn = load_graph(d, sl)
        q = 0.5 * (g.ndata["drug_u_emb"][tn] + g.ndata["drug_v_emb"][tn])
        y = g.ndata["label"][tn].numpy()
        pr = []
        for i in range(0, q.shape[0], 8192):
            pr.append((F.normalize(q[i:i + 8192], dim=-1).to(device) @ Pn.T).argmax(1).cpu())
        p = torch.cat(pr).numpy()
        np.savez_compressed(os.path.join(d.pred, f"preds_CHEMBERTA_{sl}.npz"), y_true=y, y_pred=p)
        res4[sl] = full_metrics(y, p, C, train_counts, TAIL_THRESHOLDS)
        print(f"  {sl:<11} acc={res4[sl]['acc']:.4f} strict={res4[sl]['macro_f1_strict']:.4f}")
        del g
        gc.collect()
    json.dump(res4, open(os.path.join(d.out, "results_phase4.json"), "w"), indent=2)
    mark(d, "phase4")
    return res4


# ---------------------------------------------------------------------------
# retrieval-index fix for E1.4 / E2.4
# ---------------------------------------------------------------------------
@torch.no_grad()
def _evaluate_fixed(model, g, nids, C, tcounts, tag, d, device):
    model.eval().to(device)
    dl = DataLoader(g, nids, NeighborSampler(FANOUTS), batch_size=1024, shuffle=False, num_workers=0)
    y, p = collect_preds(model, dl)
    np.savez_compressed(os.path.join(d.pred_fixed, f"preds_{tag}.npz"), y_true=y, y_pred=p)
    return full_metrics(y, p, C, tcounts, TAIL_THRESHOLDS)


def fix_e1_4(d, device):
    """Re-evaluate the E1.4 checkpoint with a retrieval index over each union
    evaluation graph (candidates still = training nodes). Inference only."""
    meta = pickle.load(open(d.meta, "rb"))
    C, train_counts = meta["C"], meta["train_counts"]
    g_train, _ = load_graph(d, "train")
    n_train = g_train.num_nodes()
    del g_train
    cache_path = os.path.join(d.ckpt, f"knn_cache_k{PROTO_K}.pt")   # as in the thesis script
    cached = None
    if os.path.exists(cache_path):
        c = torch.load(cache_path, map_location="cpu")
        cached = (c["knn_idx"], c["knn_sim"])
    ck = os.path.join(d.ckpt, "ckpt_E1_4_set_ldam_proto_best.pt")
    results = {}
    for group, members in SLICE_GROUPS.items():
        print(f"\n  --- {group}: building index on the union node set ---")
        g_ref, _ = load_graph(d, members[0])
        X, ki, ks = build_union_index(g_ref, n_train, PROTO_K, cached, device=device)
        rare_union = (train_counts[g_ref.ndata["label"]] < RARE_THRESHOLD)
        print(f"    rare nodes in union graph: {int(rare_union.sum()):,} of {rare_union.numel():,}")
        torch.save({"knn_idx": ki, "knn_sim": ks}, os.path.join(d.ckpt, f"knn_union_{group}_k{PROTO_K}.pt"))
        del g_ref
        gc.collect()
        for sl in members:
            g, tn = load_graph(d, sl)
            if tn is None:
                print(f"    {sl}: no test-node file, skipped")
                continue
            proto = SimilarityPrototype(X, ki, ks, tau=PROTO_TAU)
            model = GraphSAGEBlock(X.shape[1], C, C, 0.1, fusion_type="set", proto_module=proto,
                                   proto_fusion_mode="alpha", proto_apply="rare_only",
                                   rare_node_mask=rare_union)
            load_checkpoint(model, ck, drop_graph_buffers=True)
            m = _evaluate_fixed(model, g, tn, C, train_counts, f"E1_4_FIXED_{sl}", d, device)
            m["oob_lookups"], m["augmented_lookups"] = model._oob, model._aug
            results[sl] = m
            print(f"    {sl:<11} strict={m['macro_f1_strict']:.4f} tail15={m['tail_f1_thr15']:.4f} "
                  f"acc={m['acc']:.4f} | oob={model._oob:,} augmented={model._aug:,}")
            del model, proto, g
            gc.collect()
        del X, ki, ks
        gc.collect()
    json.dump(results, open(os.path.join(d.out, "results_E1_4_fixed.json"), "w"), indent=2)
    return results


def fix_e2_4(d, device):
    meta = pickle.load(open(d.meta, "rb"))
    C = meta["C"]
    ctl_counts = pickle.load(open(d.ctl_meta, "rb"))["train_counts"]
    g_ctl_tr, _ = load_graph(d, "ctl_train")
    n_ctl = g_ctl_tr.num_nodes()
    del g_ctl_tr
    ck = os.path.join(d.ckpt, "ckpt_E2_4_set_ldam_proto_best.pt")
    g_ct, tn_ct = load_graph(d, "ctl_test")
    X2, ki2, ks2 = build_union_index(g_ct, n_ctl, PROTO_K, None, device=device)
    rare2 = (ctl_counts[g_ct.ndata["label"]] < RARE_THRESHOLD)
    print(f"    rare nodes: {int(rare2.sum()):,} of {rare2.numel():,}")
    proto2 = SimilarityPrototype(X2, ki2, ks2, tau=PROTO_TAU)
    model2 = GraphSAGEBlock(X2.shape[1], C, C, 0.1, fusion_type="set", proto_module=proto2,
                            proto_fusion_mode="alpha", proto_apply="rare_only", rare_node_mask=rare2)
    load_checkpoint(model2, ck, drop_graph_buffers=True)
    m2 = _evaluate_fixed(model2, g_ct, tn_ct, C, ctl_counts, "E2_4_FIXED_test", d, device)
    m2["oob_lookups"], m2["augmented_lookups"] = model2._oob, model2._aug
    print(f"    E2.4 fixed: strict={m2['macro_f1_strict']:.4f} tail15={m2['tail_f1_thr15']:.4f} "
          f"head15={m2['head_f1_thr15']:.4f} acc={m2['acc']:.4f} | oob={model2._oob:,} "
          f"augmented={model2._aug:,}")
    json.dump(m2, open(os.path.join(d.out, "results_E2_4_fixed.json"), "w"), indent=2)
    return m2


# ---------------------------------------------------------------------------
# phase 5: tables  using the fixed E1.4 / E2.4 rows as in the thesis
# ---------------------------------------------------------------------------
def phase5_tables(d):
    def jload(n):
        p = os.path.join(d.out, n)
        return json.load(open(p)) if os.path.exists(p) else {}
    r2, r3, r4 = jload("results_phase2.json"), jload("results_phase3.json"), jload("results_phase4.json")
    f14, f24 = jload("results_E1_4_fixed.json"), jload("results_E2_4_fixed.json")
    if f14 and "E1_4_set_ldam_proto" in r2:
        r2 = dict(r2)
        r2["E1_4_set_ldam_proto (fixed index)"] = {**r2["E1_4_set_ldam_proto"], **f14}
    if f24 and "E2_4_set_ldam_proto" in r3:
        r3 = dict(r3)
        r3["E2_4_set_ldam_proto (fixed index)"] = {**r3["E2_4_set_ldam_proto"], "test": f24}

    print("\nTABLE 8.3 - cold-start macro-F1 (STRICT, all classes)")
    print(f"  {'run':<36}{'val':>8}{'ep':>5}" + "".join(f"{s:>12}" for s in SLICES))
    for r, row in r2.items():
        print(f"  {r:<36}{row['best_val_strict']:>8.4f}{row['best_epoch']:>5}"
              + "".join(f"{row[s]['macro_f1_strict']:>12.4f}" for s in SLICES if s in row))
    print("\nTABLE 8.4 - tail-class F1 (threshold 15)")
    for r, row in r2.items():
        print(f"  {r:<36}" + "".join(f"{row[s]['tail_f1_thr15']:>12.4f}" for s in SLICES if s in row))
    print("\nTABLE 8.5 - warm-start control")
    print(f"  {'run':<36}{'macroF1':>10}{'tail15':>10}{'head15':>10}{'acc':>10}")
    for r, row in r3.items():
        m = row["test"]
        print(f"  {r:<36}{m['macro_f1_strict']:>10.4f}{m['tail_f1_thr15']:>10.4f}"
              f"{m['head_f1_thr15']:>10.4f}{m['acc']:>10.4f}")
    print("\nTABLE 8.6 - ChemBERTa nearest-prototype floor")
    for s, m in r4.items():
        print(f"  {s:<12} strict={m['macro_f1_strict']:.4f} acc={m['acc']:.4f}")
    return r2, r3, r4
