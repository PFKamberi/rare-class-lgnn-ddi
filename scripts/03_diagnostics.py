"""Step 3: warm-start diagnostics."""
import os

import numpy as np
import torch

from _common import parser, save_json, setup, warm_graph
from ddi.experiments import diagnostics as D


def main():
    p = parser(__doc__)
    p.add_argument("--thesis-checkpoints", action="store_true",
                   help="use the thesis checkpoints and kNN cache (full data only)")
    args = p.parse_args()
    paths, device, tag = setup(args)
    if args.thesis_checkpoints and (args.test or args.sample):
        raise SystemExit("--thesis-checkpoints needs the full graph (drop --test/--sample)")

    g, num_classes = warm_graph(paths, args, tag)
    if args.thesis_checkpoints:
        ck, knn_cache, suffix = paths.thesis_warm_ckpt_dir, paths.thesis_knn_cache_k5, "_thesis"
    else:
        ck = os.path.join(paths.warm_dir, f"checkpoints_{tag}")
        knn_cache = os.path.join(paths.warm_dir, f"knn_{tag}", "knn_cache_k5.pt")
        suffix = ""
    for f in (os.path.join(ck, "ckpt_baseline.pt"), os.path.join(ck, "ckpt_proto_concat_rare_only_k5.pt")):
        if not os.path.exists(f):
            raise SystemExit(f"{f} not found: run 02_train_warmstart.py first")

    ctrl = D.control_logits(g, os.path.join(ck, "ckpt_baseline.pt"),
                            os.path.join(ck, "ckpt_proto_concat_rare_only_k5.pt"),
                            device, knn_cache_path=knn_cache)
    np.savez(os.path.join(paths.diag_dir, f"control_preds_{tag}{suffix}.npz"), **ctrl)
    train_counts, tail = D.train_counts_and_tail(g, num_classes, thr=15)
    print(f"rare classes: {int(tail.sum())} | rare test interactions: {int(tail[ctrl['y_true']].sum())}")

    print("\n== Table 7.10: control table")
    t710 = D.control_table(ctrl, tail, num_classes)
    print("\n== Table 7.9: META-DDIE frequency groups")
    t79 = D.frequency_groups(ctrl, num_classes)
    cache = torch.load(knn_cache, map_location="cpu")
    knn_idx, knn_sim = cache["knn_idx"], cache["knn_sim"]
    print("\n== Table 7.11: label dependence")
    t711 = D.label_dependence(ctrl, g, knn_idx, knn_sim, train_counts)
    print("\n== kNN label purity")
    purity = D.knn_purity(g, knn_idx, knn_sim)
    save_json({"table_7_10": t710, "table_7_9": t79,
               "table_7_11": {str(k): v for k, v in t711.items()}, "knn_purity": purity},
              os.path.join(paths.diag_dir, f"diagnostics_{tag}{suffix}.json"))


if __name__ == "__main__":
    main()
