"""Step 2: warm start (Chapter 7). Trains the five configurations and evaluates them
with five strict draws each (Table 7.8)."""
import gc
import os
import shutil

import torch

from _common import parser, save_json, setup, warm_graph
from ddi import evaluate, train


def main():
    p = parser(__doc__)
    p.add_argument("--skip-train", action="store_true", help="only evaluate existing checkpoints")
    p.add_argument("--epochs", type=int, default=None, help="default: 100 (2 with --test)")
    p.add_argument("--eval-thesis-checkpoints", action="store_true",
                   help="evaluate the original thesis checkpoints instead (full data only)")
    args = p.parse_args()
    paths, device, tag = setup(args)
    if args.eval_thesis_checkpoints and (args.test or args.sample):
        raise SystemExit("--eval-thesis-checkpoints needs the full graph (drop --test/--sample)")

    g, num_classes = warm_graph(paths, args, tag)
    print("classes:", num_classes)
    ckpt_dir = os.path.join(paths.warm_dir, f"checkpoints_{tag}")
    knn_dir = os.path.join(paths.warm_dir, f"knn_{tag}")
    os.makedirs(knn_dir, exist_ok=True)

    if not args.skip_train and not args.eval_thesis_checkpoints:
        run_args = dict(train.COMMON_RUN_ARGS)
        run_args["num_epochs"] = args.epochs or (2 if args.test else run_args["num_epochs"])
        results = {}
        for name, kw in train.WARM_START_VARIANTS.items():
            k = kw.get("proto_k")
            cache = os.path.join(knn_dir, f"knn_cache_k{k}.pt") if k else None
            _, m = train.run_variant(g, name, checkpoint_dir=ckpt_dir, knn_cache_path=cache,
                                     device=device, **run_args, **kw)
            results[name] = m
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        save_json(results, os.path.join(paths.warm_dir, f"train_log_metrics_{tag}.json"))

    eval_dir, suffix = ckpt_dir, ""
    if args.eval_thesis_checkpoints:
        eval_dir, suffix = paths.thesis_warm_ckpt_dir, "_thesis"
        target = os.path.join(knn_dir, "knn_cache_k5.pt")
        if os.path.exists(paths.thesis_knn_cache_k5) and not os.path.exists(target):
            shutil.copy(paths.thesis_knn_cache_k5, target)
    evaluate.evaluate_checkpoint_dir(
        g, eval_dir, n_draws=5, device=device, knn_cache_dir=knn_dir,
        out_json=os.path.join(paths.warm_dir, f"table_7_8_{tag}{suffix}.json"))
    print("Thesis Table 7.8: full model (concat, k=5) macro-F1 0.814 +/- 0.004, rare-F1 0.705; "
          "baseline 0.729 / 0.369")


if __name__ == "__main__":
    main()
