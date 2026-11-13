"""
Retrieval augmentation (a.k.a. Rare-Class Feature-Augmentationd).
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def build_topk_cosine_knn(X_base, candidate_ids, k=20, device="cuda", chunk=4096, exclude_self=True):
    """
    Top-k cosine neighbours for every node in X_base, retrieving only from
    candidate_ids (the training nodes, so no label leakage).
    Returns knn_idx [N, k] (global ids, CPU) and knn_sim [N, k] (CPU).
    """
    X = X_base.to(device)
    X = F.normalize(X, dim=-1)
    cand = candidate_ids.to(device)
    Xc = X[cand]
    N = X.shape[0]
    knn_idx = torch.empty((N, k), dtype=torch.long, device="cpu")
    knn_sim = torch.empty((N, k), dtype=torch.float32, device="cpu")

    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        Q = X[start:end]
        sim = Q @ Xc.T
        vals, pos = torch.topk(sim, k=k, dim=1)
        idx = cand[pos]
        if exclude_self:
            # KNOWN ISSUE (kept): a self-match is replaced by the k-th
            # neighbour, so that neighbour is counted twice (the original
            # comment called this "a quick fix").
            q_ids = torch.arange(start, end, device=device).unsqueeze(1)
            mask_self = (idx == q_ids)
            if mask_self.any():
                idx = idx.clone()
                vals = vals.clone()
                idx[mask_self] = idx[:, -1:].expand_as(idx)[mask_self]
                vals[mask_self] = vals[:, -1:].expand_as(vals)[mask_self]
        knn_idx[start:end] = idx.detach().cpu()
        knn_sim[start:end] = vals.detach().cpu()
    return knn_idx, knn_sim


class SimilarityPrototype(nn.Module):
    """Similarity-weighted mean of a node's k retrieved training neighbours."""
    def __init__(self, X_base, knn_idx, knn_sim, tau=0.07, weighting="softmax"):
        super().__init__()
        self.register_buffer("X_base", X_base)
        self.register_buffer("knn_idx", knn_idx)
        self.register_buffer("knn_sim", knn_sim)
        self.tau = tau
        self.weighting = weighting

    def forward(self, node_ids):
        device = node_ids.device
        idx = self.knn_idx[node_ids].to(device)
        sim = self.knn_sim[node_ids].to(device)
        if self.weighting == "softmax":
            w = F.softmax(sim / self.tau, dim=-1)
        elif self.weighting == "relu_norm":
            w = torch.clamp(sim, min=0.0)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")
        neigh = self.X_base[idx].to(device)
        return torch.sum(w.unsqueeze(-1) * neigh, dim=1)


class PrototypeFusion(nn.Module):
    """
    Fuse the interaction representation h with its retrieved context p.
      alpha  : a*h + (1-a)*p, one learned scalar  ("additive" in the thesis)
      gate   : per-dimension gate from [h, p]     ("gated")
      concat : MLP on [h, p]                      ("concatenation", the full model)
      attn   : 2-token attention over [h, p]
    """
    def __init__(self, dim, mode="alpha", num_heads=4, hidden_mult=2, dropout=0.1, alpha_init=0.7):
        super().__init__()
        self.mode = mode
        self.dim = dim
        if mode == "alpha":
            self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha_init)))
        elif mode == "gate":
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.Sigmoid())
        elif mode == "concat":
            self.mlp = nn.Sequential(
                nn.Linear(dim * 2, hidden_mult * dim), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_mult * dim, dim))
        elif mode == "attn":
            assert dim % num_heads == 0, "dim must be divisible by num_heads"
            self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                              dropout=dropout, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))
            self.ln = nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

    def forward(self, h, p):
        if self.mode == "alpha":
            a = torch.sigmoid(self.alpha_logit)
            return a * h + (1 - a) * p
        if self.mode == "gate":
            g = self.gate(torch.cat([h, p], dim=-1))
            return g * h + (1 - g) * p
        if self.mode == "concat":
            return self.mlp(torch.cat([h, p], dim=-1))
        if self.mode == "attn":
            tokens = torch.stack([h, p], dim=1)
            out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
            z = out[:, 0, :]
            return self.ln(z + self.ffn(z))
        raise RuntimeError("Unreachable")


def build_retrieval_x_base(g, num_classes, proto_k=5, proto_tau=0.07, proto_weighting="softmax",
                           rare_threshold=15, device="cuda", knn_cache_path=None):
    """
    Retrieval index over the mean of the two frozen ChemBERTa endpoint vectors,
    candidates = training nodes. Returns (SimilarityPrototype, rare_node_mask).
    rare_node_mask marks nodes whose class has < rare_threshold training examples.
    """
    if knn_cache_path is None:
        knn_cache_path = f"knn_cache_k{proto_k}.pt"

    X_base = 0.5 * (g.ndata['drug_u_emb'] + g.ndata['drug_v_emb'])
    X_base = X_base.detach().cpu().float()

    train_mask = g.ndata["train_mask"].bool()
    train_ids = train_mask.nonzero(as_tuple=False).squeeze(1).cpu()

    train_labels = g.ndata["label"][train_mask].cpu()
    class_counts = torch.bincount(train_labels, minlength=num_classes)
    rare_classes = class_counts < rare_threshold
    rare_node_mask = rare_classes[g.ndata["label"].cpu()]

    if os.path.exists(knn_cache_path):
        cache = torch.load(knn_cache_path, map_location="cpu")
        knn_idx, knn_sim = cache["knn_idx"], cache["knn_sim"]
        if knn_idx.shape != (g.num_nodes(), proto_k):
            raise ValueError(
                f"kNN cache {knn_cache_path} has shape {tuple(knn_idx.shape)}, expected "
                f"({g.num_nodes()}, {proto_k}). It was built for a different graph; "
                f"point knn_cache_path at a file for this graph.")
        print(f"Loaded kNN cache from {knn_cache_path}: {knn_idx.shape}")
    else:
        print("Building kNN cache (top-k cosine over candidate=train nodes)...")
        knn_idx, knn_sim = build_topk_cosine_knn(
            X_base=X_base, candidate_ids=train_ids, k=proto_k,
            device=device, chunk=4096, exclude_self=True)
        os.makedirs(os.path.dirname(knn_cache_path) or ".", exist_ok=True)
        torch.save({"knn_idx": knn_idx, "knn_sim": knn_sim}, knn_cache_path)
        print(f"Saved kNN cache to {knn_cache_path}: {knn_idx.shape}")

    proto_module = SimilarityPrototype(
        X_base=X_base, knn_idx=knn_idx, knn_sim=knn_sim,
        tau=proto_tau, weighting=proto_weighting).to(device)
    return proto_module, rare_node_mask


# ---------------------------------------------------------------------------
# cold start: index over a union (train + held-out) graph
# ---------------------------------------------------------------------------
@torch.no_grad()
def knn_rows(X_all, query_ids, cand_ids, k=5, chunk=2048, device="cuda"):
    """Cosine top-k of query rows against the candidate pool. No self-exclusion."""
    Xn = F.normalize(X_all.to(device), dim=-1)
    Xc = Xn[cand_ids.to(device)]
    cand_dev = cand_ids.to(device)
    nq = query_ids.numel()
    ki = torch.empty((nq, k), dtype=torch.long)
    ks = torch.empty((nq, k), dtype=torch.float32)
    t0 = time.time()
    for s0 in range(0, nq, chunk):
        e = min(nq, s0 + chunk)
        q = Xn[query_ids[s0:e].to(device)]
        vals, pos = torch.topk(q @ Xc.T, k=k, dim=1)
        ki[s0:e] = cand_dev[pos].cpu()
        ks[s0:e] = vals.cpu()
        if (s0 // chunk) % 20 == 0:
            frac = e / nq
            el = time.time() - t0
            print(f"      {e:>8,}/{nq:,}  {100*frac:5.1f}%  eta {el/max(frac,1e-9)-el:6.0f}s", flush=True)
    del Xn, Xc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ki, ks


def build_union_index(union_graph, n_train, proto_k=5, cached_train_knn=None, device="cuda"):
    """Full-node index for a union graph whose first n_train nodes are the
    training nodes. Training rows are copied from the training-time cache when
    its row count matches; otherwise recomputed.

    KNOWN ISSUE (kept): recomputed training rows take the top k+1 and drop rank
    0 (the self-match), whereas build_topk_cosine_knn replaces a self-match with
    the k-th neighbour. In the thesis run the training-time cache was stored
    under a different file name, so the fix script recomputed the training rows
    with this second rule.
    """
    N = union_graph.num_nodes()
    X = 0.5 * (union_graph.ndata["drug_u_emb"] + union_graph.ndata["drug_v_emb"])
    X = X.detach().cpu().float()
    ki = torch.zeros((N, proto_k), dtype=torch.long)
    ks = torch.zeros((N, proto_k), dtype=torch.float32)

    if cached_train_knn is not None and cached_train_knn[0].shape[0] == n_train:
        print(f"    reusing {n_train:,} cached training rows")
        ki[:n_train], ks[:n_train] = cached_train_knn
    else:
        print(f"    recomputing {n_train:,} training rows (cache absent or mismatched)")
        tri = torch.arange(n_train)
        a, b = knn_rows(X, tri, tri, k=proto_k + 1, device=device)
        ki[:n_train], ks[:n_train] = a[:, 1:], b[:, 1:]

    held = torch.arange(n_train, N)
    if held.numel():
        print(f"    computing {held.numel():,} held-out rows")
        a, b = knn_rows(X, held, torch.arange(n_train), k=proto_k, device=device)
        ki[n_train:], ks[n_train:] = a, b
    return X, ki, ks
