"""Step 1: extract the thesis graph and, with --test, build the small test graphs.

The thesis graph cannot be regenerated (its sampled non-interacting pairs were drawn
unseeded), so it is always taken from ddi_graph_and_linegraph.zip in the data root.
"""
import os

import dgl
import torch

from _common import parser, setup, test_drug_graph_path, test_pair_path
from ddi import graph as G
from ddi.data import save_graph_pair


def main():
    p = parser(__doc__)
    p.add_argument("--seed", type=int, default=0, help="seed for the test-subgraph drug sample")
    args = p.parse_args()
    if args.sample:
        raise SystemExit("--sample does not use the thesis graph: run build_sample_graph.py instead")
    paths, _, _ = setup(args)

    G.ensure_graph_bin(paths.graph_zip, paths.graph_bin)
    print(f"thesis graph: {paths.graph_bin} ({os.path.getsize(paths.graph_bin)/1e9:.2f} GB)")
    if not args.test:
        return

    pair = test_pair_path(paths, args.n_test_drugs)
    cold = test_drug_graph_path(paths, args.n_test_drugs)
    if os.path.exists(pair) and os.path.exists(cold):
        print("test graphs already present")
        sub, lg = G.load_graph_pair(pair)
    else:
        drug_graph = G.load_drug_graph(paths.graph_bin)      # drug graph only: a few GB of RAM
        print(drug_graph)
        sub, lg = G.make_test_subgraph(drug_graph, n_drugs=args.n_test_drugs, seed=args.seed)
        save_graph_pair(pair, sub, lg)
        # cold start uses the graph without the sampled non-interacting pairs
        sub_cs, _ = G.make_test_subgraph(drug_graph, n_drugs=args.n_test_drugs, seed=args.seed,
                                         drop_negatives=True)
        dgl.save_graphs(cold, [sub_cs])
    for f in (pair, cold):
        print(f"  {f}  {os.path.getsize(f)/1e6:.1f} MB")

    counts = torch.bincount(sub.edata["label"])
    present = (counts > 0).nonzero().flatten()
    print(f"classes present: {len(present)} | with >= 4 examples: {int((counts >= 4).sum())} | "
          f"label-0 pairs: {int(counts[0])} of {sub.num_edges()}")
    print(f"test line graph: {lg.num_nodes():,} nodes, {lg.num_edges():,} edges")


if __name__ == "__main__":
    main()
