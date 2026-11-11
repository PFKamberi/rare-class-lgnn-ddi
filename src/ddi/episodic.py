"""
Episodic META-DDIE-protocol evaluation (Tables 7.12 and 7.13).

Source: metaddie_clean_run_CLAUDE.ipynb cells 12 (training), 17 (episodes),
19 (T3 ChemBERTa-only baseline), 22 (T4 inference-time smoothing).

Changes from the notebook (device only): the notebook moved the whole graph to
the GPU (g.to(device)) and sampled there, which needs CUDA DGL. Here the graph
and sampling stay on the CPU and the model moves features; embeddings Z are
still stored on the GPU.

KNOWN ISSUES (kept):
  * episode seeds use hash(split_name) % 7. Python randomises string hashes per
    process (PYTHONHASHSEED), so episode draws differ between sessions even
    with SEED fixed. Set PYTHONHASHSEED before starting Python to pin them.
  * hidden width = in_feats (384) here, unlike warm start (212).
  * early stopping uses sklearn-default macro-F1 on the common validation set.
"""

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from dgl.dataloading import DataLoader, NeighborSampler
from tqdm.auto import tqdm

from .losses import LDAMLossNormalized
from .model import GraphSAGEBlock, SetTransformerFusion
from .splits import COMMON_MIN, COMMON_MAX, FEWER_MIN, FEWER_MAX, RARE_MIN, RARE_MAX

NUM_COMMON_CLASSES = COMMON_MAX - COMMON_MIN + 1    # 175

# META-DDIE Table 3 (accuracy), copied into the notebook output for comparison.
METADDIE_PAPER = {
    5: {"META-DDIE": {"common": 0.8379, "fewer": 0.8263, "rare": 0.7123},
        "Transf. DeepDDI": {"common": 0.7727, "fewer": 0.7464, "rare": 0.6892}},
    10: {"META-DDIE": {"common": 0.7743, "fewer": 0.7521, "rare": 0.6437},
         "Transf. DeepDDI": {"common": 0.7603, "fewer": 0.7136, "rare": 0.6163}},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _remap_common(y):
    """Common labels 1..175 -> classifier indices 0..174."""
    return y - COMMON_MIN


def make_episodic_model(in_feats, device):
    return GraphSAGEBlock(in_feats=in_feats, hidden_feats=in_feats,
                          num_classes=NUM_COMMON_CLASSES, dropout=0.1,
                          fusion_type="set", fusion_heads=4).to(device)


# ---------------------------------------------------------------------------
# pretraining on common classes (cell 12)
# ---------------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for _, _, blocks in loader:
            logits = model(blocks)
            y = _remap_common(blocks[-1].dstdata['label'])
            yt.append(y.numpy())
            yp.append(logits.argmax(1).cpu().numpy())
    yt = np.concatenate(yt)
    yp = np.concatenate(yp)
    return {"acc": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0))}


def train_common_only(g, device, checkpoint_path, epochs=100, batch_size=512, fanouts=(15, 10),
                      lr=1e-3, weight_decay=1e-4, monitor="macro_f1", patience=20, min_delta=5e-4):
    in_feats = g.ndata['drug_u_emb'].shape[1]
    model = make_episodic_model(in_feats, device)

    train_nids = g.ndata['train_mask'].nonzero(as_tuple=False).squeeze()
    val_nids = g.ndata['val_mask'].nonzero(as_tuple=False).squeeze()

    train_labels_remap = _remap_common(g.ndata['label'][train_nids])
    cls_counts = torch.bincount(train_labels_remap, minlength=NUM_COMMON_CLASSES).float()
    cls_counts = cls_counts.clamp(min=1.0)
    ldam = LDAMLossNormalized(cls_counts.tolist(), max_m=0.5, s=30.0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    sampler = NeighborSampler(list(fanouts))
    train_loader = DataLoader(g, train_nids, sampler, batch_size=batch_size, shuffle=True, num_workers=0)
    train_eval_loader = DataLoader(g, train_nids, sampler, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(g, val_nids, sampler, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train={len(train_nids)}, Val={len(val_nids)}, classes={NUM_COMMON_CLASSES}")
    print(f"Monitor: val_{monitor}, patience={patience}, epochs={epochs}")

    best_score, best_epoch, no_improve = -float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for _, _, blocks in tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False):
            y = _remap_common(blocks[-1].dstdata['label'].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss = ldam(model(blocks), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        train_m = evaluate(model, train_eval_loader, device)
        val_m = evaluate(model, val_loader, device)
        score = val_m[monitor]
        improved = score > best_score + min_delta
        print(f"[epoch {epoch:03d}] loss={total_loss/max(n_batches,1):.4f} | "
              f"train acc={train_m['acc']:.4f} f1={train_m['macro_f1']:.4f} | "
              f"val acc={val_m['acc']:.4f} f1={val_m['macro_f1']:.4f} | "
              f"best={best_score:.4f}@{best_epoch} no_improve={no_improve}/{patience}")
        if improved:
            best_score, best_epoch, no_improve = score, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "best_val_acc": val_m['acc'], "best_val_macro_f1": val_m['macro_f1'],
                        "monitor": monitor}, checkpoint_path)
            print(f"  saved new best to {checkpoint_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stop at epoch {epoch} (best val_{monitor}={best_score:.4f} @ {best_epoch})")
                break

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"loaded best checkpoint: epoch {ckpt['epoch']}, val_acc={ckpt['best_val_acc']:.4f}, "
          f"val_macro_f1={ckpt['best_val_macro_f1']:.4f}")
    return model


def load_pretrained(g, checkpoint_path, device):
    model = make_episodic_model(g.ndata['drug_u_emb'].shape[1], device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_acc={ckpt['best_val_acc']:.4f}, "
          f"val_macro_f1={ckpt['best_val_macro_f1']:.4f}")
    return model


# ---------------------------------------------------------------------------
# episodes (cell 17)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_all_embeddings(model, g, device, fanouts=(10, 10), batch_size=2048):
    sampler = NeighborSampler(list(fanouts))
    all_nids = torch.arange(g.num_nodes())
    loader = DataLoader(g, all_nids, sampler, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=0)
    model.eval()
    Z = None
    for _, output_nodes, blocks in tqdm(loader, desc="embedding", leave=False):
        emb = model.forward_embed(blocks)
        if Z is None:
            Z = torch.zeros(g.num_nodes(), emb.size(1), device=device)
        Z[output_nodes.to(device)] = emb
    return Z


def build_label_to_nodes(nids, labels_all):
    d = defaultdict(list)
    for nid in nids.cpu().numpy().tolist():
        d[int(labels_all[nid].item())].append(int(nid))
    return d


def eligible(label_to_nodes, N, Q):
    need = 1 + Q
    elig = [lbl for lbl, nodes in label_to_nodes.items() if len(nodes) >= need]
    if len(elig) < N:
        raise ValueError(f"Only {len(elig)} eligible labels; need {N}.")
    return elig


@torch.no_grad()
def run_prototype_episodes(Z, labels_all, pool, N, Q, episodes, seed):
    """1-shot nearest-prototype (cosine to the single support embedding)."""
    rng = random.Random(seed)
    dev = Z.device
    elig = eligible(pool, N, Q)
    yt, yp = [], []
    for _ in tqdm(range(episodes), desc=f"proto N={N}", leave=False):
        chosen = rng.sample(elig, N)
        s_ids, q_ids = [], []
        for lbl in chosen:
            picks = rng.sample(pool[lbl], 1 + Q)
            s_ids.append(picks[0])
            q_ids.extend(picks[1:])
        sup = torch.tensor(s_ids, device=dev)
        qry = torch.tensor(q_ids, device=dev)
        sims = F.normalize(Z[qry], dim=1) @ F.normalize(Z[sup], dim=1).T
        pred_idx = sims.argmax(dim=1).cpu().numpy()
        yp.append(np.array([chosen[i] for i in pred_idx]))
        yt.append(labels_all[qry].cpu().numpy().astype(int))
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return {"acc": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0))}


def run_linhead_episodes(Z, labels_all, pool, N, Q, episodes, seed, steps=200, lr=0.05):
    """1-shot linear head fitted on the N support embeddings per episode."""
    rng = random.Random(seed)
    dev = Z.device
    d = Z.size(1)
    elig = eligible(pool, N, Q)
    yt, yp = [], []
    for _ in tqdm(range(episodes), desc=f"linhead N={N}", leave=False):
        chosen = rng.sample(elig, N)
        s_ids, q_ids = [], []
        for lbl in chosen:
            picks = rng.sample(pool[lbl], 1 + Q)
            s_ids.append(picks[0])
            q_ids.extend(picks[1:])
        Xs = Z[torch.tensor(s_ids, device=dev)].detach()
        Xq = Z[torch.tensor(q_ids, device=dev)].detach()
        ys = torch.arange(N, device=dev)
        head = nn.Linear(d, N).to(dev)
        opt = torch.optim.SGD(head.parameters(), lr=lr)
        head.train()
        for _ in range(steps):
            loss = F.cross_entropy(head(Xs), ys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pred_idx = head(Xq).argmax(dim=1).cpu().numpy()
        yp.append(np.array([chosen[i] for i in pred_idx]))
        yt.append(labels_all[torch.tensor(q_ids, device=dev)].cpu().numpy().astype(int))
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return {"acc": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0))}


def make_pools(g, device):
    """common = test portion of common classes; fewer/rare = every node of those classes."""
    labels_all_dev = g.ndata['label'].long().to(device)
    common_test_nids = torch.where(
        g.ndata['test_mask'].bool().to(device)
        & (labels_all_dev >= COMMON_MIN) & (labels_all_dev <= COMMON_MAX))[0]
    fewer_nids = torch.where((labels_all_dev >= FEWER_MIN) & (labels_all_dev <= FEWER_MAX))[0]
    rare_nids = torch.where((labels_all_dev >= RARE_MIN) & (labels_all_dev <= RARE_MAX))[0]
    pools = {
        "common": build_label_to_nodes(common_test_nids, labels_all_dev),
        "fewer": build_label_to_nodes(fewer_nids, labels_all_dev),
        "rare": build_label_to_nodes(rare_nids, labels_all_dev),
    }
    return labels_all_dev, pools


def run_episodic_suite(Z, labels_all_dev, pools, episodes, seed, ways=(5, 10), seed_salt=""):
    """All (pool, N) combinations; skips pools with too few eligible classes.
    seed_salt reproduces the notebook's seed formula for T3/T4 (hash(split + mode))."""
    results = {}
    for split_name, pool in pools.items():
        for N_way in ways:
            try:
                eligible(pool, N_way, Q=1)
            except ValueError as e:
                print(f"[skip] {split_name} N={N_way}: {e}")
                continue
            proto = run_prototype_episodes(
                Z, labels_all_dev, pool, N=N_way, Q=1, episodes=episodes,
                seed=seed + 1000 * N_way + hash(split_name + seed_salt) % 7)
            lin = run_linhead_episodes(
                Z, labels_all_dev, pool, N=N_way, Q=1, episodes=episodes,
                seed=seed + 2000 * N_way + hash(split_name + seed_salt) % 7)
            results[(split_name, N_way, "prototype")] = proto
            results[(split_name, N_way, "linear_head")] = lin
            print(f"{split_name:6s} N={N_way:<2d} | proto acc={proto['acc']:.4f} f1={proto['macro_f1']:.4f}"
                  f" | linhead acc={lin['acc']:.4f} f1={lin['macro_f1']:.4f}")
    return results


# ---------------------------------------------------------------------------
# T3: ChemBERTa-only baseline (cell 19)
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_chemberta_only_embeddings(g, device, fusion_mode="mean", seed=42):
    """Fused drug-pair embeddings with no GNN: mean, hadamard, or an untrained
    SetTransformerFusion."""
    h_u = g.ndata['drug_u_emb'].to(device)
    h_v = g.ndata['drug_v_emb'].to(device)
    if fusion_mode == "mean":
        Z = 0.5 * (h_u + h_v)
    elif fusion_mode == "hadamard":
        Z = h_u * h_v
    elif fusion_mode == "set_untrained":
        torch.manual_seed(seed)
        fusion = SetTransformerFusion(dim=h_u.shape[1], num_heads=4).to(device)
        fusion.eval()
        outs = []
        BATCH = 65536
        for start in range(0, h_u.size(0), BATCH):
            end = min(start + BATCH, h_u.size(0))
            outs.append(fusion(h_u[start:end], h_v[start:end]))
        Z = torch.cat(outs, dim=0)
    else:
        raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
    return Z


# ---------------------------------------------------------------------------
# T4: inference-time smoothing of the frozen embeddings (cell 22)
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_knn_to_train(Z, train_nids, k=5, chunk=8192):
    """Top-k cosine neighbours of every node among train_nids (no self-exclusion)."""
    device = Z.device
    N, d = Z.shape
    Zn = F.normalize(Z, dim=-1)
    train_Zn = Zn[train_nids]
    print(f"kNN: querying {N} nodes against {len(train_nids)} training neighbors, k={k}")
    knn_idx = torch.empty((N, k), dtype=torch.long, device=device)
    knn_sim = torch.empty((N, k), dtype=torch.float32, device=device)
    for start in tqdm(range(0, N, chunk), desc=f"kNN k={k}", leave=False):
        end = min(N, start + chunk)
        sims = Zn[start:end] @ train_Zn.T
        vals, pos = torch.topk(sims, k=k, dim=1)
        knn_idx[start:end] = train_nids[pos]
        knn_sim[start:end] = vals
    return knn_idx, knn_sim


def compute_smoothed_Z(Z, knn_idx, knn_sim, apply_mask, fusion_mode="concat", tau=0.07, fusion_params=None):
    """Parameter-free smoothing (alpha / replace / mean) on apply_mask nodes only."""
    w = F.softmax(knn_sim / tau, dim=-1)
    neigh = Z[knn_idx]
    protos = (w.unsqueeze(-1) * neigh).sum(dim=1)
    if fusion_mode == "alpha":
        a = fusion_params.get("alpha", 0.7) if fusion_params else 0.7
        fused = a * Z + (1 - a) * protos
    elif fusion_mode == "replace":
        fused = protos
    elif fusion_mode == "mean":
        fused = 0.5 * Z + 0.5 * protos
    else:
        raise ValueError(f"fusion_mode={fusion_mode} needs trained parameters; use alpha, replace or mean.")
    out = Z.clone()
    out[apply_mask] = fused[apply_mask]
    return out


T4_FUSION_CONFIGS = [
    ("alpha_a0.7", "alpha", {"alpha": 0.7}),
    ("alpha_a0.5", "alpha", {"alpha": 0.5}),
    ("alpha_a0.9", "alpha", {"alpha": 0.9}),
    ("replace", "replace", {}),
]


def results_to_json(results):
    return {"_".join(map(str, k[:-2])) + f"_N{k[-2]}_{k[-1]}": v for k, v in results.items()}
