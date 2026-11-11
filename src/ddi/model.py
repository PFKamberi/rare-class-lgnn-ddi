"""
Model: symmetric endpoint fusion -> (optional) retrieval augmentation ->
two GraphSAGE layers -> linear classifier.

Sources:
  fusion modules, GraphSAGEBlock   THIS_IS_THE_ONE_I_RUN cell 15 (warm start)
  ConcatFusion, out-of-range guard ch8_rebuild_resumable (local) cell 4
  forward_embed                    metaddie_clean_run_CLAUDE cell 10
  ScatterSAGEConv                  ch8_rebuild_resumable (local) cell 4

One GraphSAGEBlock serves all four experiment families. Its parameter names are
the same as in each thesis notebook, so every thesis checkpoint loads into it.

The only deliberate change from the warm-start and episodic notebooks is the
message-passing layer: dgl.nn.SAGEConv(aggregator_type="mean") is replaced by
ScatterSAGEConv, which computes the same thing in plain PyTorch. This lets DGL
stay on the CPU (graph building and neighbour sampling only), so one code path
runs on Colab, an A100 and the RTX 5090. Because DGL is CPU-only, sampled
blocks are no longer moved to the GPU; forward() moves the features instead.
tests/test_scatter_sage.py checks the layer against DGL's SAGEConv.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from .retrieval import PrototypeFusion


# ---------------------------------------------------------------------------
# message passing
# ---------------------------------------------------------------------------
class ScatterSAGEConv(nn.Module):
    """Pure-torch equivalent of dgl.nn.SAGEConv(in, out, aggregator_type="mean"):

        h_dst_new = fc_self(h_dst) + fc_neigh(mean_{src -> dst} h_src)

    Destinations with no sampled neighbour get a zero neighbour term, as in DGL.
    Parameter names (fc_self.weight, fc_self.bias, fc_neigh.weight) match DGL's
    SAGEConv, so DGL-trained checkpoints load without renaming.
    """
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.fc_self = nn.Linear(in_feats, out_feats, bias=True)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=False)

    def forward(self, block, x):
        num_dst = block.num_dst_nodes()
        src, dst = block.edges()
        dev = x.device
        src = src.to(dev, non_blocking=True)
        dst = dst.to(dev, non_blocking=True)
        msg = x[src]
        agg = torch.zeros(num_dst, x.size(1), device=dev, dtype=x.dtype)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(num_dst, device=dev, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=dev, dtype=x.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(1)
        return self.fc_self(x[:num_dst]) + self.fc_neigh(agg / deg)


# ---------------------------------------------------------------------------
# endpoint fusion
# ---------------------------------------------------------------------------
class SymmetricGatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, h_i, h_j):
        prod = h_i * h_j
        diff = torch.abs(h_i - h_j)
        g = self.gate(torch.cat([prod, diff], dim=-1))
        return g * (h_i + h_j) + (1 - g) * diff


class SymmetricAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scorer = nn.Linear(dim, 1)

    def forward(self, h_i, h_j):
        prod = h_i * h_j
        alpha = torch.sigmoid(self.scorer(prod))
        return alpha * (h_i + h_j) + (1 - alpha) * prod


class SetTransformerFusion(nn.Module):
    """Learned query attending over the unordered set {h_i, h_j}."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.out = nn.Linear(dim, dim)

    def forward(self, h_i, h_j):
        x = torch.stack([h_i, h_j], dim=1)
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return self.out(out.squeeze(1))


class SymmetricNTNFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim))
        self.V = nn.Linear(dim, dim)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, h_i, h_j):
        bi1 = (h_i @ self.W * h_j).sum(dim=1, keepdim=True)
        bi2 = (h_j @ self.W * h_i).sum(dim=1, keepdim=True)
        out = self.V(h_i + h_j) + bi1 + bi2 + self.b
        return torch.tanh(out)


class ConcatFusion(nn.Module):
    """Order-sensitive. Cold-start run E1.5 only."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, a, b):
        return self.proj(torch.cat([a, b], -1))


FUSIONS = {
    "gated": SymmetricGatedFusion,
    "attn": SymmetricAttentionFusion,
    "ntn": SymmetricNTNFusion,
    "concat": ConcatFusion,
}


# ---------------------------------------------------------------------------
# full model
# ---------------------------------------------------------------------------
class GraphSAGEBlock(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden_feats,
        num_classes,
        dropout=0.1,
        fusion_type="gated",
        fusion_heads=4,
        # retrieval augmentation ("prototype smoothing" in the code)
        proto_module=None,
        proto_fusion_mode="alpha",
        proto_apply="all",          # "all" or "rare_only"
        rare_node_mask=None,        # BoolTensor [N] on CPU, indexed by line-graph node id
        proto_fusion_kwargs=None,
    ):
        super().__init__()
        if fusion_type == "set":
            self.fusion = SetTransformerFusion(in_feats, num_heads=fusion_heads)
        elif fusion_type in FUSIONS:
            self.fusion = FUSIONS[fusion_type](in_feats)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.proto = proto_module
        self.proto_apply = proto_apply
        if rare_node_mask is not None:
            # KNOWN ISSUE (kept, disclosed in the thesis): the mask is derived
            # from each node's TRUE label, so routing uses label information.
            self.register_buffer("rare_node_mask", rare_node_mask.bool())
        else:
            self.rare_node_mask = None

        if proto_fusion_kwargs is None:
            proto_fusion_kwargs = {}
        self.proto_fuse = (PrototypeFusion(in_feats, mode=proto_fusion_mode, **proto_fusion_kwargs)
                           if proto_module is not None else None)

        self.layer1 = ScatterSAGEConv(in_feats, hidden_feats)
        self.layer2 = ScatterSAGEConv(hidden_feats, hidden_feats)
        self.norm1 = nn.LayerNorm(hidden_feats)
        self.norm2 = nn.LayerNorm(hidden_feats)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_feats, num_classes)

        # diagnostics from the cold-start notebook: lookups whose node id is
        # outside the retrieval index (should be 0) and augmented lookups
        self._oob = 0
        self._aug = 0

    def _input(self, blocks):
        dev = next(self.parameters()).device
        h_u = blocks[0].srcdata["drug_u_emb"].to(dev, non_blocking=True)
        h_v = blocks[0].srcdata["drug_v_emb"].to(dev, non_blocking=True)
        x_src = self.fusion(h_u, h_v)

        if self.proto is not None:
            src_nids = blocks[0].srcdata[dgl.NID].to(dev, non_blocking=True)  # ids in the full graph
            n = self.proto.knn_idx.shape[0]
            oob = src_nids >= n
            if oob.any():
                self._oob += int(oob.sum())
            safe = src_nids.clamp(max=n - 1)
            p = self.proto(safe)
            if self.proto_apply == "rare_only" and (self.rare_node_mask is not None):
                m = self.rare_node_mask[safe].to(dev) & (~oob)
                if m.any():
                    self._aug += int(m.sum())
                    fused = self.proto_fuse(x_src[m], p[m])
                    x_src = x_src.clone()
                    x_src[m] = fused
            else:
                x_src = self.proto_fuse(x_src, p)
        return x_src

    def _embed(self, blocks):
        x = self._input(blocks)
        h = self.layer1(blocks[0], x)
        h = F.relu(self.norm1(h))
        h = self.dropout(h)
        h = self.layer2(blocks[1], h)
        h = F.relu(self.norm2(h))
        h = self.dropout(h)
        return h

    def forward(self, blocks):
        return self.classifier(self._embed(blocks))

    def forward_embed(self, blocks):
        """Pre-classifier embedding (used by the episodic evaluation)."""
        return self._embed(blocks)


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
GRAPH_SPECIFIC_BUFFERS = ("rare_node_mask", "proto.X_base", "proto.knn_idx", "proto.knn_sim")


def state_dict_from_checkpoint(obj):
    """Thesis checkpoints come in three shapes: {"model_state_dict": ...}
    (warm start, episodic), {"model": ...} (cold start) or a raw state dict
    (final_model_*.pt)."""
    if isinstance(obj, dict):
        for k in ("model_state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def load_checkpoint(model, path, device="cpu", drop_graph_buffers=False, strict=False):
    """Load a thesis checkpoint. drop_graph_buffers=True ignores the stored
    retrieval index and rare mask (needed when evaluating on a graph whose node
    set differs from the training graph, as in the cold-start fix)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = state_dict_from_checkpoint(ck)
    if drop_graph_buffers:
        sd = {k: v for k, v in sd.items() if k not in GRAPH_SPECIFIC_BUFFERS}
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    real_missing = [k for k in missing if k not in GRAPH_SPECIFIC_BUFFERS]
    if real_missing or unexpected:
        print(f"  WARNING missing={real_missing} unexpected={list(unexpected)}")
    return ck
