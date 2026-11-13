import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.dataloading import DataLoader, NeighborSampler
from sklearn.metrics import f1_score

from ..model import GraphSAGEBlock, load_checkpoint
from ..retrieval import build_retrieval_x_base

GROUPS = {"common (#1-#175)": (1, 175),
          "fewer (#176-#204)": (176, 204),
          "rare (#205-#227)": (205, 227)}


def train_counts_and_tail(g, num_classes, thr=15):
    tm = g.ndata["train_mask"].bool()
    tc = torch.bincount(g.ndata["label"][tm], minlength=num_classes)
    return tc, (tc < thr).numpy()


# ---------------------------------------------------------------------------
# control logits
# ---------------------------------------------------------------------------
@torch.no_grad()
def logits_for(model, g, test_nids, seed=0, batch_size=512):
    dgl.seed(seed)
    torch.manual_seed(seed)
    dl = DataLoader(g, test_nids, NeighborSampler([15, 10]), batch_size=batch_size,
                    shuffle=False, num_workers=0)
    model.eval()
    ys, zs = [], []
    for _, _, blocks in dl:
        zs.append(model(blocks).cpu())
        ys.append(blocks[-1].dstdata["label"].cpu())
    return torch.cat(ys).numpy(), torch.cat(zs).numpy()


def control_logits(g, base_ckpt, smooth_ckpt, device, knn_cache_path=None, rare_thr=15, seed=0):
    """Baseline and full-model (concat, k=5) logits on the test split, one draw each."""
    num_classes = int(g.ndata["label"].max().item() + 1)
    in_feats = g.ndata["drug_u_emb"].shape[1]
    test_nids = g.ndata["test_mask"].nonzero(as_tuple=False).squeeze(1)

    base = GraphSAGEBlock(in_feats, num_classes, num_classes, fusion_type="set").to(device)
    load_checkpoint(base, base_ckpt, device=device)
    yt_b, z_b = logits_for(base, g, test_nids, seed)
    del base

    proto, rare_mask = build_retrieval_x_base(g, num_classes, proto_k=5, proto_tau=0.07,
                                              proto_weighting="softmax", rare_threshold=rare_thr,
                                              device=device, knn_cache_path=knn_cache_path)
    full = GraphSAGEBlock(in_feats, num_classes, num_classes, fusion_type="set",
                          proto_module=proto, proto_fusion_mode="concat",
                          proto_apply="rare_only", rare_node_mask=rare_mask).to(device)
    load_checkpoint(full, smooth_ckpt, device=device)
    yt_s, z_s = logits_for(full, g, test_nids, seed)
    expected = g.ndata["label"][test_nids].numpy()
    assert np.array_equal(yt_b, expected) and np.array_equal(yt_s, expected), \
        "predictions are not in test_nids order"
    return dict(y_true=yt_b, logits_base=z_b, y_true_s=yt_s, logits_smooth=z_s)


def control_table(ctrl, tail, num_classes):
    """Table 7.10: baseline / baseline restricted to rare classes ("rarity hint")
    / full model."""
    yt_b, z_b = ctrl["y_true"], ctrl["logits_base"]
    yt_s, z_s = ctrl["y_true_s"], ctrl["logits_smooth"]
    z_hint = z_b.copy()
    z_hint[np.nonzero(tail[yt_b])[0][:, None], np.nonzero(~tail)[0][None, :]] = -np.inf
    all_lbl = np.arange(num_classes)
    rows = []
    print(f"{'model':<26}{'tailF1':>9}{'macroEvt':>10}{'rare correct':>15}")
    for name, yt, yp in [("baseline", yt_b, z_b.argmax(1)),
                         ("+ rarity hint", yt_b, z_hint.argmax(1)),
                         ("+ prototype smoothing", yt_s, z_s.argmax(1))]:
        pc = f1_score(yt, yp, labels=all_lbl, average=None, zero_division=0)
        is_rare = tail[yt]
        ok, n = int((yt[is_rare] == yp[is_rare]).sum()), int(is_rare.sum())
        rows.append(dict(model=name, tailF1=float(pc[tail].mean()), macroEvt=float(pc[1:].mean()),
                         rare_correct=ok, n_rare=n))
        print(f"{name:<26}{pc[tail].mean():>9.4f}{pc[1:].mean():>10.4f}{ok:>10}/{n}")
    return rows


# ---------------------------------------------------------------------------
# frequency groups 
# ---------------------------------------------------------------------------
def frequency_groups(ctrl, num_classes):
    y = ctrl["y_true"]
    out = {}
    for name, key in (("baseline", "logits_base"), ("full model", "logits_smooth")):
        p = ctrl[key].argmax(1)
        print(f"\n{name}")
        print(f"  {'group':<22}{'classes':>9}{'test n':>9}{'macroF1':>10}")
        out[name] = {}
        for gname, (lo, hi) in GROUPS.items():
            ids = np.arange(lo, min(hi, num_classes - 1) + 1)
            n = int(np.isin(y, ids).sum())
            f1 = f1_score(y, p, average="macro", labels=ids, zero_division=0) if len(ids) else 0.0
            out[name][gname] = dict(classes=int(len(ids)), test_n=n, macro_f1=float(f1))
            print(f"  {gname:<22}{len(ids):>9}{n:>9}{f1:>10.4f}")
    return out


# ---------------------------------------------------------------------------
# label dependence
# ---------------------------------------------------------------------------
def true_class_rank(z, y):
    z_true = z[np.arange(len(y)), y][:, None]
    return (z > z_true).sum(1)


def label_dependence(ctrl, g, knn_idx, knn_sim, train_counts, thresholds=(15, 20), tau=0.07,
                     n_boot=10000, seed=0):
    """Table 7.11: rare test nodes split by whether their retrieved neighbours
    contain a same-class training interaction (group A) or not (group B)."""
    from scipy.stats import fisher_exact, mannwhitneyu
    rng = np.random.default_rng(seed)
    yt_b, z_b, z_s = ctrl["y_true"], ctrl["logits_base"], ctrl["logits_smooth"]
    labels_f = g.ndata["label"]
    test_nids = g.ndata["test_mask"].nonzero(as_tuple=False).squeeze(1)
    pred_b, pred_s = z_b.argmax(1), z_s.argmax(1)
    rank_b, rank_s = true_class_rank(z_b, yt_b), true_class_rank(z_s, yt_b)

    nbr_lbl_test = labels_f[knn_idx[test_nids]].numpy()
    w_test = F.softmax(knn_sim[test_nids] / tau, dim=-1).numpy()
    same = nbr_lbl_test == yt_b[:, None]
    same_mass = (w_test * same).sum(1)
    has_same = same.any(1)
    tcn = train_counts.numpy()

    def boot_diff(a, b):
        if len(a) == 0 or len(b) == 0:
            return (float("nan"), float("nan"))
        out = np.empty(n_boot)
        for i in range(n_boot):
            out[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
        return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

    report = {}
    for THR in thresholds:
        tail = (train_counts < THR).numpy()
        rare = tail[yt_b]
        r_idx = np.nonzero(rare)[0]
        print("\n" + "=" * 78)
        print(f"RARE THRESHOLD < {THR}   ({int(tail.sum())} classes, {len(r_idx)} test nodes)")
        print("=" * 78)
        if len(r_idx) == 0:
            print("no rare test nodes")
            continue
        gA, gB = r_idx[has_same[r_idx]], r_idx[~has_same[r_idx]]
        print(f"{'group':<10}{'n':>4}{'base ok':>9}{'aug ok':>8}{'gain':>7}{'W->R':>6}{'R->W':>6}"
              f"{'rank_b':>9}{'rank_a':>9}{'d rank':>8}{'mass':>7}{'n_train':>9}")
        rows = {}
        for name, grp in (("A same", gA), ("B none", gB), ("all", r_idx)):
            if len(grp) == 0:
                print(f"{name:<10}{0:>4}")
                continue
            ok_b, ok_s = pred_b[grp] == yt_b[grp], pred_s[grp] == yt_b[grp]
            drank = (rank_b[grp] - rank_s[grp]).astype(float)
            rows[name] = dict(n=int(len(grp)), base_ok=int(ok_b.sum()), aug_ok=int(ok_s.sum()),
                              gained=int((~ok_b & ok_s).sum()), lost=int((ok_b & ~ok_s).sum()),
                              d_rank=float(drank.mean()), mass=float(same_mass[grp].mean()),
                              _ok_s=ok_s, _drank=drank, _g=grp)
            r = rows[name]
            print(f"{name:<10}{r['n']:>4}{r['base_ok']:>9}{r['aug_ok']:>8}{r['aug_ok']-r['base_ok']:>7}"
                  f"{r['gained']:>6}{r['lost']:>6}{rank_b[grp].mean():>9.1f}{rank_s[grp].mean():>9.1f}"
                  f"{r['d_rank']:>8.1f}{r['mass']:>7.3f}{tcn[yt_b[grp]].mean():>9.1f}")
        if "A same" in rows and "B none" in rows:
            A, B = rows["A same"], rows["B none"]
            tbl = np.array([[int(A["_ok_s"].sum()), int((~A["_ok_s"]).sum())],
                            [int(B["_ok_s"].sum()), int((~B["_ok_s"]).sum())]])
            _, p_f = fisher_exact(tbl)
            try:
                _, p_mw = mannwhitneyu(A["_drank"], B["_drank"], alternative="two-sided")
            except ValueError:
                p_mw = float("nan")
            lo, hi = boot_diff(A["_drank"], B["_drank"])
            print(f"\naugmented-model accuracy, A vs B: Fisher exact p = {p_f:.3f}")
            print(f"rank improvement A {A['d_rank']:+.1f} | B {B['d_rank']:+.1f} | "
                  f"diff 95% CI [{lo:+.1f}, {hi:+.1f}] | Mann-Whitney p = {p_mw:.3f}")
        report[THR] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                       for k, v in rows.items()}
    return report


# ---------------------------------------------------------------------------
# kNN label purity 
# ---------------------------------------------------------------------------
def knn_purity(g, knn_idx, knn_sim, tau=0.07):
    labels_f = g.ndata["label"]
    num_classes = int(labels_f.max().item() + 1)
    tm = g.ndata["train_mask"].bool()
    train_counts = torch.bincount(labels_f[tm], minlength=num_classes)
    neigh_lbl = labels_f[knn_idx]
    w = F.softmax(knn_sim / tau, dim=-1)
    out = {}

    def report(node_mask, name):
        idx = node_mask.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            print(f"{name}: no nodes")
            return
        own = labels_f[idx].unsqueeze(1)
        same = (neigh_lbl[idx] == own)
        r = dict(n=int(idx.numel()),
                 frac_same=float(same.float().mean(dim=1).mean()),
                 mass_same=float((w[idx] * same.float()).sum(dim=1).mean()),
                 any_same=float(same.any(dim=1).float().mean()),
                 top1_same=float(same[:, 0].float().mean()),
                 distinct=float(torch.tensor([len(torch.unique(x)) for x in neigh_lbl[idx]]).float().mean()))
        out[name] = r
        print(f"\n{name}  (n = {r['n']})")
        print(f"  neighbours sharing the node's class      : {100*r['frac_same']:.1f}%")
        print(f"  prototype MASS on same-class neighbours  : {100*r['mass_same']:.1f}%")
        print(f"  nodes with >=1 same-class neighbour      : {100*r['any_same']:.1f}%")
        print(f"  nodes whose nearest neighbour is same-cls: {100*r['top1_same']:.1f}%")
        print(f"  distinct classes among the neighbours    : {r['distinct']:.2f}")

    rare15 = (train_counts < 15)[labels_f]
    rare20 = (train_counts < 20)[labels_f]
    report(rare20, "RARE nodes (class train count < 20)")
    report(rare15, "RARE nodes (class train count < 15)")
    report(~rare20, "COMMON nodes (not smoothed)")
    p = train_counts.double() / train_counts.sum()
    out["chance"] = float((p ** 2).sum())
    print(f"\nchance rate (two random training interactions share a class): {100*out['chance']:.2f}%")
    return out
