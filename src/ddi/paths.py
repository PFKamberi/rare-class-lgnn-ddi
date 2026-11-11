"""
All file locations in one place.

Every path the thesis notebooks hard-coded (``/content/drive/MyDrive/...`` on
Colab, ``/home/ppcrniai/Documents/thesis_ch8/...`` on the 5090) is built here
from two roots, so the notebooks never contain a literal path:

    DATA_ROOT  where the inputs live   (Drive on Colab, a local folder otherwise)
    WORK_ROOT  where outputs are written (checkpoints, caches, predictions, JSON)

The notebooks call ``get_paths(DATA_ROOT, WORK_ROOT)`` with the two values set in
their first cell. Nothing else in the package reads the environment.
"""

from dataclasses import dataclass, asdict
import os

# Google Drive id of the raw META-DDIE data zip (Drug_META_DDIE.db,
# event_META_DDIE_5foldLabel.db). Same id in every thesis notebook.
RAW_DATA_GDRIVE_ID = "1K-5lQI2Dy5dyp2SrY8-QoaWGm0fgyVNP"

# Placeholder strings. Replace them in the first cell of each notebook.
DATA_ROOT_PLACEHOLDER = "<DATA_ROOT: e.g. /content/drive/MyDrive on Colab>"
WORK_ROOT_PLACEHOLDER = "<WORK_ROOT: e.g. /content/drive/MyDrive/ddi_runs on Colab>"


@dataclass
class Paths:
    data_root: str
    work_root: str

    # ---- inputs ------------------------------------------------------------
    raw_data_dir: str          # holds Drug_META_DDIE.db, event_META_DDIE_5foldLabel.db
    drug_db: str
    event_db: str
    graph_zip: str             # ddi_graph_and_linegraph.zip (thesis warm-start/episodic graph)
    graph_bin: str             # where the zip is extracted to
    # thesis artefacts that already exist (read-only, for re-evaluation)
    thesis_warm_ckpt_dir: str  # thesis_smart_fusion_whole_graph_checkpoints
    thesis_hidden384_dir: str  # thesis_hidden384_experiment
    thesis_knn_cache_k5: str   # knn_cache_k5.pt
    thesis_episodic_dir: str   # thesis_metaddie_clean_run
    thesis_ch8_dir: str        # the folder holding graphs_v4/, checkpoints_v4/, results_v4/

    # ---- outputs (everything this repo writes) -----------------------------
    test_graph_dir: str        # small subgraphs for smoke tests
    warm_dir: str
    diag_dir: str
    episodic_dir: str
    coldstart_dir: str

    def make_output_dirs(self):
        for p in (self.work_root, self.test_graph_dir, self.warm_dir, self.diag_dir,
                  self.episodic_dir, self.coldstart_dir):
            os.makedirs(p, exist_ok=True)

    def show(self):
        for k, v in asdict(self).items():
            flag = "" if (os.path.exists(v) or not os.path.isabs(v)) else "   (does not exist yet)"
            print(f"  {k:<22} {v}{flag}")


def get_paths(data_root: str, work_root: str, local_bin_dir: str = None) -> Paths:
    """Build every path from the two roots.

    data_root     : folder containing the thesis inputs. On Colab this is
                    /content/drive/MyDrive (the thesis notebooks kept everything
                    at the top level of MyDrive).
    work_root     : folder for new outputs. Keep it separate from data_root so
                    nothing written here can overwrite a thesis artefact.
    local_bin_dir : where to extract the graph zip. On Colab use /content
                    (local disk, fast); defaults to work_root.
    """
    for r in (data_root, work_root):
        if r.startswith("<"):
            raise ValueError(f"Placeholder not replaced: {r}")
    local_bin_dir = local_bin_dir or work_root
    raw = os.path.join(data_root, "data")
    return Paths(
        data_root=data_root,
        work_root=work_root,
        raw_data_dir=raw,
        drug_db=os.path.join(raw, "Drug_META_DDIE.db"),
        event_db=os.path.join(raw, "event_META_DDIE_5foldLabel.db"),
        graph_zip=os.path.join(data_root, "ddi_graph_and_linegraph.zip"),
        graph_bin=os.path.join(local_bin_dir, "ddi_graph_and_linegraph.bin"),
        thesis_warm_ckpt_dir=os.path.join(data_root, "thesis_smart_fusion_whole_graph_checkpoints"),
        thesis_hidden384_dir=os.path.join(data_root, "thesis_hidden384_experiment"),
        thesis_knn_cache_k5=os.path.join(data_root, "knn_cache_k5.pt"),
        thesis_episodic_dir=os.path.join(data_root, "thesis_metaddie_clean_run"),
        thesis_ch8_dir=os.path.join(data_root, "thesis_ch8"),
        test_graph_dir=os.path.join(work_root, "test_graphs"),
        warm_dir=os.path.join(work_root, "warmstart"),
        diag_dir=os.path.join(work_root, "diagnostics"),
        episodic_dir=os.path.join(work_root, "episodic"),
        coldstart_dir=os.path.join(work_root, "coldstart"),
    )
