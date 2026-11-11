"""
Loading the thesis graph and building the small test subgraph.

Sources:
  * load/attach: THIS_IS_THE_ONE_I_RUN.ipynb cells 6-9 (identical code in
    metaddie_clean_run cell 4 and rerun_hidden384 cell 2).
  * make_test_subgraph: new, for smoke tests only. Nothing in the thesis used it.
"""

import os
import zipfile

import torch
import dgl


def ensure_graph_bin(graph_zip: str, graph_bin: str):
    """Extract ddi_graph_and_linegraph.bin from the zip if it is not there yet."""
    if os.path.exists(graph_bin):
        return graph_bin
    os.makedirs(os.path.dirname(graph_bin) or ".", exist_ok=True)
    print(f"unzipping {graph_zip} -> {os.path.dirname(graph_bin)}")
    with zipfile.ZipFile(graph_zip) as z:
        names = [n for n in z.namelist() if n.endswith("ddi_graph_and_linegraph.bin")]
        assert names, "ddi_graph_and_linegraph.bin not found inside the zip"
        z.extract(names[0], os.path.dirname(graph_bin))
        extracted = os.path.join(os.path.dirname(graph_bin), names[0])
        if extracted != graph_bin:
            os.replace(extracted, graph_bin)
    return graph_bin


def load_drug_graph(graph_bin: str):
    """Load ONLY the drug-level graph (index 0). Cheap: does not read the
    640M-edge line graph, so it needs only a few GB of RAM."""
    graphs, _ = dgl.load_graphs(graph_bin, [0])
    return graphs[0]


def load_graph_pair(graph_bin: str):
    """Load [drug graph, line graph]. The full thesis line graph needs a
    machine with about 64 GB of RAM."""
    graphs, _ = dgl.load_graphs(graph_bin)
    subgraph, line_subgraph = graphs
    return subgraph, line_subgraph


def attach_endpoint_features(subgraph, line_subgraph):
    """Line-graph node i is drug-graph edge i: copy its endpoint embeddings
    and its label onto the line-graph node (cell 9, verbatim)."""
    u, v = subgraph.edges()
    drug_feats = subgraph.ndata['smiles_embedding']
    line_subgraph.ndata['drug_u_emb'] = drug_feats[u]
    line_subgraph.ndata['drug_v_emb'] = drug_feats[v]
    line_subgraph.ndata['label'] = subgraph.edata['label']
    return line_subgraph


def make_test_subgraph(drug_graph, n_drugs=800, seed=0, drop_negatives=False):
    """Random drug subset -> induced drug subgraph -> directed line graph.

    Keeps every interaction between the sampled drugs (and, unless
    drop_negatives=True, every sampled non-interacting pair among them), then
    applies the same line-graph rule as the thesis (backtracking=False).
    Rare classes mostly vanish at this size; the subgraph only checks that the
    pipeline runs end to end.

    drop_negatives=True removes the label-0 pairs, which gives the no-negative
    drug graph the cold-start chapter is built on.
    """
    g = torch.Generator().manual_seed(seed)
    n_drugs = min(n_drugs, drug_graph.num_nodes())
    nodes = torch.randperm(drug_graph.num_nodes(), generator=g)[:n_drugs].sort().values
    sub = dgl.node_subgraph(drug_graph, nodes)
    # keep the original ids under plain names; drop DGL's bookkeeping keys so
    # later subgraph calls do not overwrite them
    sub.ndata['orig_drug_id'] = sub.ndata.pop(dgl.NID)
    sub.edata['orig_edge_id'] = sub.edata.pop(dgl.EID)
    if drop_negatives:
        keep = (sub.edata['label'] != 0).nonzero(as_tuple=False).squeeze(1)
        sub = dgl.edge_subgraph(sub, keep, relabel_nodes=False)   # edge order preserved
        sub.edata.pop(dgl.EID, None)
        sub.ndata.pop(dgl.NID, None)
    lg = dgl.line_graph(sub, backtracking=False)
    print(f"test subgraph: {sub.num_nodes()} drugs, {sub.num_edges():,} drug-graph edges "
          f"-> line graph {lg.num_nodes():,} nodes, {lg.num_edges():,} edges")
    return sub, lg
