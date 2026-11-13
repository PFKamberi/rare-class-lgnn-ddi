"""Step 5: cold start (Chapter 8). Drug-disjoint split, Setup A/B graphs, runs E1.1-E1.5
and E2.1-E2.4, the ChemBERTa floor, the retrieval-index fix for E1.4/E2.4, Tables 8.1-8.7.
Every phase caches its output and skips work already on disk."""
import os
import shutil

import dgl

from _common import parser, setup, test_drug_graph_path
from ddi.experiments import coldstart as CS


def main():
    p = parser(__doc__)
    p.add_argument("--thesis-parent-graph", default=None,
                   help="the thesis graphs_v4/parent_graph.bin, to reproduce the thesis drug split "
                        "(default: <data root>/thesis_ch8/graphs_v4/parent_graph.bin if present)")
    args = p.parse_args()
    paths, device, tag = setup(args)
    tag = tag if args.test else "v4"
    d = CS.ColdStartDirs(base=paths.coldstart_dir, tag=tag)
    thesis_parent = args.thesis_parent_graph or os.path.join(paths.thesis_ch8_dir, "graphs_v4", "parent_graph.bin")

    CS.banner("PHASE 0 - drug-disjoint parent graph")
    if os.path.exists(d.parent):
        print("  cached")
    elif args.test:
        path = test_drug_graph_path(paths, args.n_test_drugs)
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found: run 01_prepare_data.py --test first")
        CS.phase0_from_drug_graph(d, dgl.load_graphs(path)[0][0])
    elif os.path.exists(thesis_parent):
        shutil.copy(thesis_parent, d.parent)
        print(f"  copied the thesis parent graph (thesis drug split) from {thesis_parent}")
    else:
        from ddi.data import download_raw_data
        from ddi.paths import RAW_DATA_GDRIVE_ID
        drug_db, event_db = download_raw_data(paths.drug_db, paths.event_db, RAW_DATA_GDRIVE_ID)
        print("  WARNING: fresh rebuild from the databases; the drug split will NOT match the thesis")
        CS.phase0_build_parent(d, drug_db, event_db)
    CS.mark(d, "phase0")
    CS.phase0_report(d)

    epochs = dict(max_epochs=2, min_epochs=1) if args.test else {}
    CS.banner("PHASE 1 - class filter and line graphs")
    CS.phase1(d)
    CS.mark(d, "phase1")
    CS.banner("PHASE 2 - cold-start runs E1.1-E1.5")
    CS.phase2(d, device, **epochs)
    CS.banner("PHASE 3 - warm-start control E2.1-E2.4")
    CS.phase3(d, device, **epochs)
    CS.banner("PHASE 4 - ChemBERTa nearest-prototype floor")
    CS.phase4(d, device)
    CS.banner("FIX E1.4")
    CS.fix_e1_4(d, device)
    CS.banner("FIX E2.4")
    CS.fix_e2_4(d, device)
    CS.banner("TABLES")
    CS.phase5_tables(d)
    CS.mark(d, "phase5")
    print("\nThesis Table 8.3, E1.3 (Setup A/B, CS1 then CS2): 0.251 0.334 0.055 0.079")


if __name__ == "__main__":
    main()
