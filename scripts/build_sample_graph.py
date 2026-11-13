"""Build a small sample graph from the public META-DDIE data, so the pipeline can run
without the thesis graph.

1. Downloads the META-DDIE databases (the authors' Google Drive link, as used in the
   thesis) into <data root>/data, unless they are already there.
2. Takes a class-stratified sample of the interactions (event types 1..227): at most
   --per-class interactions per event type, so the common types are capped and the
   fewer (176-204) and rare (205-227) types are kept whole.
3. Builds the drug graph with ChemBERTa-77M embeddings (downloaded from Hugging Face),
   adds sampled non-interacting pairs (class 0) as the thesis graph has them, and builds
   the directed line graph with the thesis rule.
4. Saves [drug graph, line graph] and the drug graph without class 0 (for cold start)
   under <work root>/sample/. The other scripts read them with --sample.

The sample is NOT the thesis graph: results on it show the method running on real
interactions at a small scale, not the thesis numbers.
"""
import os

import dgl
import numpy as np
import torch

from _common import parser, sample_dir, sample_drug_graph_path, sample_pair_path, setup
from ddi import data as D
from ddi.paths import RAW_DATA_GDRIVE_ID
from ddi.splits import COMMON_MAX, COMMON_MIN, FEWER_MAX, FEWER_MIN, RARE_MAX, RARE_MIN

MAX_LABEL = 227          # event types >= 228 have one interaction each (as in META-DDIE)


def stratified_sample(events, per_class, seed):
    """At most `per_class` interactions of every event type; smaller types are kept whole."""
    rng = np.random.RandomState(seed)
    keep = []
    for _, idx in sorted(events.groupby("label").indices.items()):
        if len(idx) > per_class:
            idx = np.sort(rng.choice(idx, per_class, replace=False))
        keep.append(idx)
    return events.iloc[np.concatenate(keep)].reset_index(drop=True)


def add_negatives(g, ratio, seed):
    """Append int(ratio * #interactions) random drug pairs with label 0. Seeded; no
    self-loops, no pair that interacts in either direction, no duplicates."""
    gen = torch.Generator().manual_seed(seed)
    u, v = g.edges()
    taken = set(zip(u.tolist(), v.tolist())) | set(zip(v.tolist(), u.tolist()))
    n_neg, n = int(g.num_edges() * ratio), g.num_nodes()
    src, dst = [], []
    while len(src) < n_neg:
        a = torch.randint(0, n, (2 * n_neg,), generator=gen).tolist()
        b = torch.randint(0, n, (2 * n_neg,), generator=gen).tolist()
        for x, y in zip(a, b):
            if x != y and (x, y) not in taken:
                taken.add((x, y))
                taken.add((y, x))
                src.append(x)
                dst.append(y)
                if len(src) == n_neg:
                    break
    g.add_edges(torch.tensor(src), torch.tensor(dst),
                data={"label": torch.zeros(n_neg, dtype=torch.long),
                      "id": torch.arange(g.num_edges(), g.num_edges() + n_neg)})
    return g


def group_counts(labels):
    lab = torch.as_tensor(labels)
    out = {}
    for name, (lo, hi) in (("common", (COMMON_MIN, COMMON_MAX)), ("fewer", (FEWER_MIN, FEWER_MAX)),
                           ("rare", (RARE_MIN, RARE_MAX))):
        m = (lab >= lo) & (lab <= hi)
        out[name] = (int(torch.unique(lab[m]).numel()), int(m.sum()))
    return out


def build(paths, per_class=200, neg_ratio=0.3, seed=0):
    drug_db, event_db = D.download_raw_data(paths.drug_db, paths.event_db, RAW_DATA_GDRIVE_ID)
    events = D.get_all_events(event_db).reset_index(drop=True)
    events["label"] = events["label"].astype(int)
    events = events[events.label <= MAX_LABEL].reset_index(drop=True)
    sample = stratified_sample(events, per_class, seed)
    print(f"interactions: {len(events):,} in the database, {len(sample):,} in the sample "
          f"(at most {per_class} per event type)")

    g = D.graph_construction(D.load_data_into_dataframe(drug_db, "drug"), sample)   # runs ChemBERTa
    g.edata.pop("fused_embedding", None)
    drug_graph_noneg = dgl.graph(g.edges(), num_nodes=g.num_nodes())
    drug_graph_noneg.ndata["smiles_embedding"] = g.ndata["smiles_embedding"]
    drug_graph_noneg.edata["label"] = g.edata["label"].clone()
    g = add_negatives(g, neg_ratio, seed)
    lg = dgl.line_graph(g, backtracking=False)

    os.makedirs(sample_dir(paths), exist_ok=True)
    D.save_graph_pair(sample_pair_path(paths), g, lg)
    dgl.save_graphs(sample_drug_graph_path(paths), [drug_graph_noneg])

    print(f"drugs: {g.num_nodes():,} | interactions: {drug_graph_noneg.num_edges():,} "
          f"+ {g.num_edges() - drug_graph_noneg.num_edges():,} sampled non-interacting pairs (class 0)")
    for name, (n_cls, n_int) in group_counts(drug_graph_noneg.edata["label"]).items():
        print(f"  {name:<7} {n_cls:>4} event types, {n_int:>7,} interactions")
    print(f"line graph: {lg.num_nodes():,} nodes, {lg.num_edges():,} edges")
    print(f"saved {sample_pair_path(paths)}\nsaved {sample_drug_graph_path(paths)}")
    return g, lg, drug_graph_noneg


def main():
    p = parser(__doc__)
    p.add_argument("--per-class", type=int, default=200, help="max interactions per event type (default 200)")
    p.add_argument("--neg-ratio", type=float, default=0.3, help="class-0 pairs per interaction (thesis: 0.3)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    paths, _, _ = setup(args)
    if os.path.exists(sample_pair_path(paths)) and os.path.exists(sample_drug_graph_path(paths)):
        print(f"sample graph already present in {sample_dir(paths)} (delete it to rebuild)")
        return
    build(paths, args.per_class, args.neg_ratio, args.seed)


if __name__ == "__main__":
    main()
