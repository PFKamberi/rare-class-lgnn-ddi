# ddi-linegraph

Code for the MSc thesis *Rare-Class Feature-Augmented Line-Graph Neural Networks for Drug-Drug Interaction Event Prediction under Class Imbalance* (Petros Kamberi, University of the Peloponnese & NCSR "Demokritos").

Drug-drug interaction (DDI) event prediction is cast as node classification on the line graph of the DrugBank/META-DDIE interaction graph: every interaction is a node, labelled with one of the event types. The model fuses the frozen ChemBERTa-77M embeddings of the two drugs, applies two GraphSAGE layers and an LDAM loss, and adds retrieval augmentation for rare classes.

This repository reproduces the thesis experiments with the thesis code, ported into a package and command-line scripts. It runs locally on one CUDA GPU (developed on an RTX 5090).

## Contents

```
src/ddi/            the package
  paths.py          every file location, built from a data root and a work root
  data.py           META-DDIE databases, ChemBERTa embeddings, drug graph
  graph.py          load the thesis graph, attach endpoint features, test subgraph
  splits.py         warm-start split, episodic (META-DDIE) split, save/check split files
  model.py          ScatterSAGEConv, endpoint fusion, GraphSAGEBlock, checkpoint loading
  retrieval.py      kNN index, similarity prototype, prototype fusion, union index
  losses.py         LDAM (two variants), focal loss, class weights
  metrics.py        strict metrics (reported) and sklearn-default metrics (logs only)
  train.py          warm-start training and the five Chapter 7 configurations
  evaluate.py       five-draw strict evaluation of checkpoints
  experiments/      diagnostics, episodic (META-DDIE protocol), cold start
scripts/            one script per step, run in order (see "Running"), and a demo on public data
setup/              environment setup and GPU check
tests/              CPU tests on synthetic graphs: pytest -q tests
```

## Installation

```bash
bash setup/setup_local.sh        # uv venv (Python 3.11), torch 2.7.0+cu128, DGL 2.1.0, this package
source .venv/bin/activate
python setup/verify_gpu.py       # GPU kernels + ScatterSAGEConv vs DGL's SAGEConv
```

DGL 2.1.0 comes from PyPI and is CPU-only. It is used only to build graphs and sample neighbours; the model runs on the GPU. The setup script patches DGL so it skips its graphbolt library, which this code does not use and which has no build for torch 2.7.

## Quick demo on public data

No thesis data is needed:

```bash
bash scripts/run_demo.sh --data-root ~/ddi_data --work-root ~/ddi_runs
```

The demo downloads the META-DDIE databases (from the authors' link, about 21 MB) and ChemBERTa-77M (from Hugging Face). It builds a class-stratified sample graph: at most 200 interactions per event type, so every common, fewer and rare type with data is represented, plus sampled non-interacting pairs as in the thesis graph. It then trains and evaluates the warm-start model (30 epochs) and runs the diagnostics. Results are in `<work root>/warmstart/table_7_8_sample.json` and `<work root>/diagnostics/`.

The sample is not the thesis graph, so its numbers are not comparable with the thesis tables. The other steps run on it too: `python scripts/04_episodic.py --sample ...`, `python scripts/05_coldstart.py --sample ...`. `build_sample_graph.py --help` lists the sampling options.

On the default sample, one run gave macro-F1 0.713 → 0.748 and rare-F1 0.547 → 0.730 (baseline → full model); expect similar but not identical numbers.

## Data

The scripts take two folders, as `--data-root` / `--work-root` or the environment variables `DDI_DATA_ROOT` / `DDI_WORK_ROOT`:

- **data root** (read only): the thesis inputs.
  - `ddi_graph_and_linegraph.zip`: the thesis graph (required).
  - `knn_cache_k5.pt`, `thesis_smart_fusion_whole_graph_checkpoints/`, `thesis_metaddie_clean_run/`: thesis kNN cache and checkpoints, needed only to re-evaluate the thesis models.
  - `thesis_ch8/graphs_v4/parent_graph.bin`: the thesis cold-start graph, needed to reproduce the thesis drug split.
  - `data/Drug_META_DDIE.db`, `data/event_META_DDIE_5foldLabel.db`: the META-DDIE databases, needed for the demo and for a fresh cold-start rebuild (downloaded with `gdown` from the META-DDIE authors' link if missing).
- **work root**: everything the scripts write (extracted graph, checkpoints, caches, JSON results). Keep it separate from the data root.

The thesis graph cannot be regenerated: its 333,237 sampled non-interacting pairs (class 0) were drawn unseeded. Always use the saved zip. Extract it only with `01_prepare_data.py`: plain `unzip` truncates the 7.7 GB file to 3.39 GB (missing zip64 size field), and the script checks the CRC.

## Running

```bash
export DDI_DATA_ROOT=~/ddi_data DDI_WORK_ROOT=~/ddi_runs

# 1. quick check that everything runs (800-drug test subgraph, minutes; numbers mean nothing)
bash scripts/run_all.sh --test

# 2. full experiments (about 64 GB of RAM for the full line graph; hours per step)
bash scripts/run_all.sh
```

Or run the steps one by one (each takes `--help`). The scripts replace the thesis notebooks, which are not part of this repository.

| Step | Script | Thesis output | Main options |
|---|---|---|---|
| 1 | `01_prepare_data.py` | extracts the thesis graph; `--test` also builds the test subgraphs | `--n-test-drugs` |
| 2 | `02_train_warmstart.py` | Table 7.8 (and the full-graph row of 7.7) | `--skip-train`, `--eval-thesis-checkpoints`, `--epochs` |
| 3 | `03_diagnostics.py` | Tables 7.9, 7.10, 7.11, kNN label purity | `--thesis-checkpoints` |
| 4 | `04_episodic.py` | Tables 7.12, 7.13 | `--thesis-checkpoint`, `--episodes`, `--epochs` |
| 5 | `05_coldstart.py` | Tables 8.1–8.7 | `--thesis-parent-graph` |

Step 3 needs the checkpoints of step 2. Step 5 reuses everything already on disk, so it can be restarted after an interruption. Results are written as JSON under the work root (`warmstart/`, `diagnostics/`, `episodic/`, `coldstart/`).

**Reference numbers (thesis).** Expect agreement within sampling noise, not bit for bit (see "Changes", items 2 and 9).

| Output | Thesis |
|---|---|
| Table 7.8, full model (concat, k=5) | macro-F1 0.814 ± 0.004, rare-F1 0.705 |
| Table 7.8, baseline | macro-F1 0.729, rare-F1 0.369 |
| Table 7.12, 5-way | 0.833 / 0.923 / 0.837 |
| Table 7.12, 10-way | 0.765 / 0.881 / 0.774 |
| Table 8.3, E1.3 (Setup A/B, CS1 then CS2) | 0.251 / 0.334 / 0.055 / 0.079 |

**Metrics.** Macro-F1 is always strict: over the event classes 1..211 in warm start (class 0, the sampled non-interacting pairs, is excluded), and over all remapped classes in cold start. Rare = fewer than 15 training examples. Training logs print sklearn's default macro-F1 and are not reported.

## Where each module comes from

| Module | Thesis notebook (cells) |
|---|---|
| `data.py` | `Thesis_baselline_undirected_linegraph_GraphSage` 5, 7, 9, 10, 13, 17 (built `ddi_graph_and_linegraph.bin`) |
| `graph.py` | `THIS_IS_THE_ONE_I_RUN` 6–9 (same code in `metaddie_clean_run` 4, `rerun_hidden384` 2); `make_test_subgraph` is new, for tests |
| `splits.py` | warm start: `THIS_IS_THE_ONE_I_RUN` 12; episodic: `metaddie_clean_run_CLAUDE` 6 |
| `metrics.py` | `THIS_IS_THE_ONE_I_RUN` 16; strict metrics: `rerun_hidden384` 2 and `Methodolody_claims_verifications_for_chapter_5` 33; `full_metrics`: `ch8_rebuild_resumable` 4 |
| `losses.py` | `THIS_IS_THE_ONE_I_RUN` 16; `LDAMLossNormalized`: `metaddie_clean_run_CLAUDE` 12; `class_weights`: `ch8_rebuild_resumable` 4 |
| `retrieval.py` | `THIS_IS_THE_ONE_I_RUN` 14, 18; union index: `ch8_rebuild_resumable` 22 |
| `model.py` | `THIS_IS_THE_ONE_I_RUN` 15; `ConcatFusion`, out-of-range guard, `ScatterSAGEConv`: `ch8_rebuild_resumable` 4; `forward_embed`: `metaddie_clean_run_CLAUDE` 10 |
| `train.py` | `THIS_IS_THE_ONE_I_RUN` 16, 19 |
| `evaluate.py` | `rerun_hidden384` 2, 5 |
| `experiments/diagnostics.py` | `Methodolody_claims_verifications_for_chapter_5` 5, 6, 14, 33 |
| `experiments/episodic.py` | `metaddie_clean_run_CLAUDE` 12, 17, 19, 22 |
| `experiments/coldstart.py` | `ch8_rebuild_resumable` (local, RTX 5090) 2–24 |

Notebook names are spelled as the original files. Tables 7.1–7.6 (design phase, subset graph) are not ported.

## Changes from the thesis code

None of these changes the method.

1. **Message passing.** `dgl.nn.SAGEConv(aggregator_type="mean")` is replaced by `ScatterSAGEConv`: the same computation in plain PyTorch, with the same parameter names, so the thesis checkpoints load unchanged. The cold-start code already used it. A test checks it against DGL (max difference about 5e-7).
2. **Device.** DGL runs on the CPU (graph building and neighbour sampling); the model runs on the GPU. The thesis warm-start and episodic notebooks sampled on the GPU, so the sampling random streams differ.
3. **Paths and caches.** All locations come from `paths.py`. A kNN cache built for a different graph raises an error instead of being used.
4. **One canonical split.** All warm-start analyses use `splits.filter_and_split_graph`. One thesis script retyped it with `test_size=0.30` instead of `1 - 0.70`, which moves one node between train and validation.
5. **Table 7.8** evaluates the baseline with five draws, like the other rows (the thesis used one draw for that row).
6. **Cold start.** The retrieval-index fix for E1.4/E2.4 (the rows the thesis reports) is part of the pipeline.
7. **Bugs that did not affect the thesis run:** `create_ddi_graph` passes `event_db_path` on; `create_line_subgraph` uses its argument instead of a global.
8. **Rebuilds from the databases** (never the saved thesis files): drugs are numbered in sorted DrugBank-id order instead of a hash-seed-dependent order, and SMILES longer than 256 characters are embedded as the mean [CLS] of overlapping chunks instead of the first chunk only (31 of 3,535 drugs). A fresh cold-start rebuild therefore gives a different drug split from the thesis; `05_coldstart.py` copies the thesis parent graph when it is present.
9. **Episodic seeds** use a stable hash (`zlib.crc32`) instead of Python's per-process `hash()`, so reruns draw the same episodes. The thesis draws cannot be redrawn, so Tables 7.12–7.13 match within episode noise. Episodic early stopping uses strict validation macro-F1; all 175 common classes are in validation, so it selects the same checkpoint as the thesis.
10. **Logs and unused metrics.** The logged tail F1 uses threshold 15 (thesis: 20). Two unreported metrics that averaged over all 212 classes including class 0 were removed. No reported number changes.

## Known issues kept on purpose

These produced the thesis numbers and are left as they were.

- The line graph is **directed** (`backtracking=False`): it keeps only head-to-tail adjacency, 39% of co-incident pairs.
- The thesis graph contains 333,237 sampled non-interacting pairs (class 0): drawn unseeded, checked against one orientation only, not de-duplicated. They are a training class in warm start and excluded from every metric.
- Warm-start hidden width = number of classes (212). A 384-wide rerun reached 0.829 ± 0.004 macro-F1 against 0.814.
- Retrieval is routed by `rare_node_mask`, which is derived from each node's **true** label (warm start and cold start).
- The kNN index replaces a training node's self-match with its k-th neighbour, so that neighbour is counted twice. The cold-start union index recomputes training rows with a different rule (top k+1, drop rank 0).
- Warm-start early stopping uses sklearn's default macro-F1 (reported numbers are strict).
- Two LDAM variants: unnormalised margins (warm start, cold start) and normalised margins (episodic).
- The thesis graph files use the old hash-seed node order and first-chunk SMILES embeddings (see "Changes", item 8).

## Terminology

The code keeps the thesis-code names: "prototype" / `proto_*` (e.g. `SimilarityPrototype`, `PrototypeFusion`, `proto_k`) is the retrieval augmentation described in the thesis, the softmax-weighted mean of the k nearest training interactions fused into the rare nodes' features. The names are kept because the thesis checkpoints store parameters under them (`proto_fuse.*`).

## Open issue

Table 7.9 in the thesis lists group sizes (common 216,415, fewer 176 test interactions) that this code does not reproduce. On the canonical split the warm-start test set has 166,570 common and 39 fewer interactions and no rare-group interactions (classes 203–211 are dropped from validation and test as too rare); the F1 values match. Use these sizes when reusing the table.

## Tests

```bash
python -m pytest -q tests
```

CPU only, on synthetic graphs, a few minutes; nothing is downloaded. `tests/test_scripts.py` runs all five scripts end to end in test mode; `tests/test_sample_graph.py` builds a sample graph from tiny databases and runs the scripts on it with `--sample`.
