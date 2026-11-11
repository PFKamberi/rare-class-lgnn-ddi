"""Drug node numbering must not depend on Python's string-hash seed."""
import os
import subprocess
import sys

SNIPPET = r'''
import pandas as pd, torch
import ddi.data as D
D.generate_smiles_embeddings = lambda g, df, m: [torch.zeros(1, 4) for _ in range(g.num_nodes())]
ev = pd.DataFrame({"id1": ["DB00005", "DB00001", "DB00003", "DB00002"],
                   "id2": ["DB00004", "DB00005", "DB00001", "DB00003"],
                   "label": [1, 2, 3, 4]})
g = D.graph_construction(pd.DataFrame({"id": [], "smile": []}), ev)
u, v = g.edges()
print(u.tolist(), v.tolist())
'''


def _run(seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    out = subprocess.run([sys.executable, "-c", SNIPPET], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_node_numbering_independent_of_hash_seed():
    results = {_run(seed) for seed in (0, 1, 2, 3, 4)}
    assert len(results) == 1, f"node numbering changed with PYTHONHASHSEED: {results}"


def test_node_numbering_is_sorted_drugbank_order():
    # sorted ids: DB00001->0, DB00002->1, DB00003->2, DB00004->3, DB00005->4
    assert _run(0) == "[4, 0, 2, 1] [3, 4, 0, 2]"