# ddi-linegraph

Code for the MSc thesis *Rare-Class Feature-Augmented Line-Graph Neural Networks for Drug-Drug Interaction Event Prediction under Class Imbalance* (Petros Kamberi, University of the Peloponnese & NCSR "Demokritos").

The repository is a modular port of the thesis notebooks that produced the thesis results. The implementations live in `src/ddi/`; the notebooks in `notebooks/` only load data, call the package and print tables. Version 0.1.0 reproduces the thesis code as it was, known issues included (listed below), so that the thesis numbers can be regenerated before anything is changed.

## Layout

```
src/ddi/
  paths.py        every file location, built from DATA_ROOT and WORK_ROOT
  data.py         META-DDIE databases, ChemBERTa embeddings, drug graph, negative sampling
  graph.py        load the thesis .bin, attach endpoint features, test subgraph
  splits.py       warm-start split, episodic (META-DDIE) split, save/check split files
  model.py        ScatterSAGEConv, endpoint fusion modules, GraphSAGEBlock, checkpoint loading
  retrieval.py    kNN index, similarity prototype, prototype fusion, union index (Ch8 fix)
  losses.py       LDAM (two variants), focal loss, class weights
  metrics.py      strict (reporting) and sklearn-default (logging only) metrics
  train.py        warm-start training loop and the five Chapter 7 configurations
  evaluate.py     five-draw strict evaluation of checkpoints (Table 7.8)
  experiments/    the thesis experiments, built on the modules above
    diagnostics.py  control table, frequency groups, label dependence, kNN purity
    episodic.py     common-only pretraining, 1-shot episodes, ChemBERTa-only, smoothing
    coldstart.py    drug-disjoint split, Setup A/B graphs, E1/E2 runs, ChemBERTa floor, index fix
notebooks/
  checks/          one-off verification notebooks (local_only.ipynb, ...)
  00_setup_and_test_subgraph.ipynb
  01_warmstart_retrieval.ipynb
  02_warmstart_diagnostics.ipynb
  03_episodic_metaddie.ipynb
  04_coldstart.ipynb
setup/
  setup_local.sh   uv venv, torch 2.7+cu128 + DGL 2.1 + this package (RTX 5090)
  verify_gpu.py    GPU kernel check + ScatterSAGEConv vs dgl SAGEConv check
tests/             CPU smoke tests on a synthetic graph (pytest -q tests)
```

## What reproduces what

| Thesis | Notebook | Module | Original notebook |
|---|---|---|---|
| Ch4 graph (`ddi_graph_and_linegraph.bin`) | – (uses the saved file) | `data.py`, `graph.py` | `Thesis_baselline_undirected_linegraph_GraphSage` |
| Table 7.7 (full-graph row), 7.8 | 01 | `train.py`, `evaluate.py` | `..._THIS_IS_THE_ONE_I_RUN`, `rerun_hidden384` |
| Tables 7.9, 7.10, 7.11, kNN purity | 02 | `experiments/diagnostics.py` | `Methodolody_claims_verifications_for_chapter_5` |
| Tables 7.12, 7.13 | 03 | `experiments/episodic.py` | `metaddie_clean_run_CLAUDE` |
| Tables 8.1–8.7 | 04 | `experiments/coldstart.py` | `ch8_rebuild_resumable` (local, 5090) |
| Tables 7.1–7.6 (design phase, subset graph) | not ported | – | numbers only in Thesis Update 3 |

## Data

Set two paths in the first cell of every notebook (`~` is allowed):

- `DATA_ROOT`: the folder with the thesis inputs (read only). It holds `ddi_graph_and_linegraph.zip`, `data/` with the two META-DDIE databases (`Drug_META_DDIE.db`, `event_META_DDIE_5foldLabel.db`), `knn_cache_k5.pt`, `thesis_smart_fusion_whole_graph_checkpoints/`, `thesis_metaddie_clean_run/`, and optionally `thesis_ch8/graphs_v4/parent_graph.bin`. If `data/` is missing, notebook 04 downloads it from Google Drive with `gdown` (needed only for a fresh cold-start rebuild).
- `WORK_ROOT`: where everything new is written. Keep it separate from `DATA_ROOT`; nothing here writes into a thesis folder.

The thesis graph must come from the saved `ddi_graph_and_linegraph.zip`. It cannot be regenerated: the 333,237 sampled non-interacting pairs were drawn unseeded, and its node order depended on Python's string-hash seed (new builds sort the drug ids).

## Running locally (RTX 5090, uv)

Installs torch 2.7.0+cu128 and DGL 2.1.0 from PyPI, then patches DGL so it skips graphbolt (its C++ library exists only for torch 2.0–2.2; this repo never uses it). Same setup as the Chapter 8 run. `data.dgl.ai` is not needed.

```bash
bash setup/setup_local.sh
source .venv/bin/activate
python setup/verify_gpu.py
jupyter lab            # kernel "ddi (5090)"
```

Run `notebooks/checks/local_only.ipynb` once to check the setup, then `00_setup_and_test_subgraph.ipynb`, then any of 01–04 with `TEST_MODE = True`.

Notebook 00 reads only the drug graph from the .bin (3,342 drugs, not the 640M-edge line graph), takes `N_TEST_DRUGS` random drugs (default 800) and every pair among them, and builds the directed line graph with the thesis rule. At 800 drugs this is roughly 80k interactions and about 9M line-graph edges (expected value); raise `N_TEST_DRUGS` for a heavier test. Most rare classes vanish at this size, so the test-mode numbers say nothing about the method; test mode only checks that the code runs end to end.

`TEST_MODE = False` uses the full graph and needs about 64 GB of RAM.

## Changes from the thesis notebooks

Only plumbing was changed:

1. **Message passing.** `dgl.nn.SAGEConv(aggregator_type="mean")` is replaced everywhere by `ScatterSAGEConv` (plain PyTorch, same maths, same parameter names, so thesis checkpoints load directly). The Chapter 8 run already used it. `tests/` and `setup/verify_gpu.py` check it against DGL (max difference about 5e-7).
2. **Device handling.** DGL runs on the CPU only (graph building and neighbour sampling). Blocks stay on the CPU and the model moves features to the GPU. The thesis warm-start and episodic notebooks moved blocks, or the whole graph, to the GPU, so neighbour-sampling random streams differ and numbers match the thesis within sampling noise, not bit for bit.
3. **Paths** are arguments built in `paths.py`. kNN caches are given explicit paths, and a cache built for a different graph raises an error instead of being indexed silently.
4. **Diagnostics** all use the canonical split from `splits.filter_and_split_graph`. One thesis script retyped it with `test_size=0.30` instead of `1 - 0.70`, which moves one node between train and validation.
5. **Table 7.8** evaluates the baseline with five draws like the other rows (the thesis used a single draw for that row).
6. **Cold start**: the retrieval-index fix for E1.4/E2.4 (the rows the thesis reports) is part of the pipeline, not a separate script.
7. Two plain bugs that did not affect the thesis run were corrected: `create_ddi_graph` now passes `event_db_path` on, and `create_line_subgraph` uses its argument instead of a global.
8. Two fixes that only affect graphs rebuilt from the databases, never the saved thesis files (which are what reproduce the thesis): drug node ids follow sorted DrugBank-id order instead of an order that depended on Python's string-hash seed, and SMILES longer than 256 characters are embedded as the mean [CLS] of overlapping chunks instead of the first chunk only (31 of 3,535 drugs). A fresh cold-start Phase 0 therefore gives a different drug split from the thesis; notebook 04 copies the thesis parent_graph.bin.

## Known issues kept on purpose

These produced the thesis numbers and are left as they were. Each is marked `KNOWN ISSUE` in the code.

- The line graph is **directed** (`backtracking=False`): it keeps only head-to-tail adjacency, 39% of co-incident pairs.
- Negative sampling checks only the stored orientation and does not de-duplicate (class 0 only, excluded from all metrics).
- Warm-start hidden width = number of classes (212). A 384-wide rerun reached 0.829 ± 0.004 macro-F1 against 0.814.
- Retrieval is routed by `rare_node_mask`, which is derived from each node's **true** label.
- The self-exclusion rule in the kNN index replaces a self-match with the k-th neighbour, so that neighbour is counted twice.
- Training logs and early stopping use sklearn's default macro-F1. Reported numbers use the strict convention (`evaluate.py`, `metrics.strict_metrics`, `metrics.full_metrics`).
- Two LDAM variants: unnormalised margins (warm start, cold start) and normalised margins (episodic).
- Episode seeds use `hash(split_name) % 7` and change between Python processes unless `PYTHONHASHSEED` is set.
- The thesis graph files number drugs in an order that depended on Python's string-hash seed. New builds number drugs in sorted DrugBank-id order and are deterministic, but a fresh cold-start Phase 0 therefore gives a different drug split from the thesis; notebook 04 can copy the thesis `parent_graph.bin`.
- In the cold-start index fix, training rows are recomputed with a top-(k+1)-drop-self rule instead of the training-time rule.
- The thesis graph files embed long SMILES (> 256 characters; 31 of 3,535 drugs, 0.97% of interactions) from their first 256-character chunk only. Graphs built with this code use the mean over all chunks (`ddi.data.chunk_and_embed`); shorter SMILES are unaffected.

## Open issue

Table 7.9 in the thesis lists group sizes (common 216,415, fewer 176 test interactions) that do not match the script it was traced to (166,570 and 39), although the fewer-group F1 values match. Notebook 02 prints the group table; re-derive Table 7.9 from it before reusing it.
