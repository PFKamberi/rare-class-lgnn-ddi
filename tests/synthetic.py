"""A tiny random drug graph with the same fields as the thesis .bin, for tests.

Labels are drawn so that every experiment family has something to work with:
  0           sampled non-interacting pairs (warm start / episodic)
  1..12       "common" META-DDIE classes
  180..183    "fewer" classes
  210..213    "rare" classes
"""
import torch
import dgl


def make_synthetic_drug_graph(n_drugs=150, n_edges=6000, dim=384, seed=0, with_negatives=True):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n_drugs, (n_edges,), generator=g)
    dst = torch.randint(0, n_drugs, (n_edges,), generator=g)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    pool = torch.cat([torch.arange(1, 13).repeat(30), torch.arange(180, 184).repeat(4),
                      torch.arange(210, 214).repeat(2)])
    if with_negatives:
        pool = torch.cat([pool, torch.zeros(120, dtype=torch.long)])
    labels = pool[torch.randint(0, len(pool), (len(src),), generator=g)]
    graph = dgl.graph((src, dst), num_nodes=n_drugs)
    graph.ndata["id"] = torch.arange(n_drugs)
    graph.ndata["smiles_embedding"] = torch.randn(n_drugs, dim, generator=g)
    graph.edata["label"] = labels
    graph.edata["id"] = torch.arange(graph.num_edges())
    return graph
