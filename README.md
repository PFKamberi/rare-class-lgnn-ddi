# Rare-Class Feature-Augmented Line-Graph Neural Networks for Drug-Drug Interaction Event Prediction under Class Imbalance

This repository contains the code for my MSc thesis, whose main objective is to improve the prediction of rare drug–drug interaction event types under strong class imbalance. Each interaction becomes a node of a line graph, represented by the frozen ChemBERTa-77M embeddings of its two drugs. Two GraphSAGE layers and an LDAM loss classify it, and nodes of rare event types are enriched with features retrieved from their most similar training interactions.

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
bash setup/setup_local.sh
source .venv/bin/activate
python setup/verify_gpu.py
```

## Quick demo on public data

```bash
bash scripts/run_demo.sh --data-root ~/ddi_data --work-root ~/ddi_runs
```

The demo downloads the META-DDIE databases (from the authors' link, about 21 MB; see [Data and credits](#data-and-credits)) and ChemBERTa-77M (from Hugging Face). It builds a class-stratified sample graph: at most 200 interactions per event type, so the common types are capped and all fewer and rare interactions are kept, plus sampled non-interacting pairs as in the thesis graph. It then trains and evaluates the warm-start model (30 epochs) and runs the diagnostics. Its numbers are not comparable with the thesis tables.

## Running

These runs need the thesis graph, `ddi_graph_and_linegraph.zip`, in the data root. It is not part of this repository.

```bash
export DDI_DATA_ROOT=~/ddi_data DDI_WORK_ROOT=~/ddi_runs

# quick check that everything runs (800-drug test subgraph, minutes; numbers mean nothing)
bash scripts/run_all.sh --test

# full experiments (about 64 GB of RAM; hours per step)
bash scripts/run_all.sh
```

Or run the steps one by one, `01_prepare_data.py` to `05_coldstart.py` (each takes `--help`).

## Tests

```bash
python -m pytest -q tests
```

## Data and credits

- **Dataset:** the META-DDIE databases (`Drug_META_DDIE.db`, `event_META_DDIE_5foldLabel.db`), released by the META-DDIE authors as [META-DDIE data.zip on Google Drive](https://drive.google.com/file/d/1K-5lQI2Dy5dyp2SrY8-QoaWGm0fgyVNP/view), linked from their repository [github.com/YifanDengWHU/META-DDIE](https://github.com/YifanDengWHU/META-DDIE). The demo downloads it automatically; to use it by hand, unzip it so the two files are in `<data root>/data/`.
- **Paper:** Deng et al., [META-DDIE: predicting drug–drug interaction events with few-shot learning](https://doi.org/10.1093/bib/bbab514), *Briefings in Bioinformatics* (2021).
- **Original source:** the interactions and drugs come from [DrugBank](https://go.drugbank.com/).
- **Drug embeddings:** [ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM) (DeepChem, Hugging Face).