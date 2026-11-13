"""Step 4: episodic META-DDIE protocol (Tables 7.12 and 7.13): common-only pretraining,
1-shot episodes, the ChemBERTa-only baseline (T3) and inference-time smoothing (T4)."""
import gc
import os

import torch

from _common import load_line_graph, parser, save_json, setup
from ddi.experiments import episodic as E
from ddi.splits import build_metaddie_graph, check_split, save_split

SEED = 42


def main():
    p = parser(__doc__)
    p.add_argument("--thesis-checkpoint", action="store_true",
                   help="use the thesis pretrained checkpoint instead of training (full data only)")
    p.add_argument("--episodes", type=int, default=None, help="default: 10000 (200 with --test)")
    p.add_argument("--epochs", type=int, default=None, help="default: 100 (2 with --test)")
    args = p.parse_args()
    paths, device, tag = setup(args)
    if args.thesis_checkpoint and (args.test or args.sample):
        raise SystemExit("--thesis-checkpoint needs the full graph (drop --test/--sample)")
    episodes = args.episodes or (200 if args.test else 10_000)

    lg = load_line_graph(paths, args)
    E.set_seed(SEED)
    g = build_metaddie_graph(lg, seed=SEED)
    del lg
    gc.collect()
    split_path = os.path.join(paths.episodic_dir, f"split_{tag}.pt")
    check_split(g, split_path) if os.path.exists(split_path) else save_split(g, split_path, seed=SEED)

    if args.thesis_checkpoint:
        model = E.load_pretrained(g, os.path.join(paths.thesis_episodic_dir, "pretrain_common_only.pt"), device)
    else:
        model = E.train_common_only(g, device, os.path.join(paths.episodic_dir, f"pretrain_common_only_{tag}.pt"),
                                    epochs=args.epochs or (2 if args.test else 100), batch_size=512,
                                    fanouts=(15, 10), monitor="macro_f1", patience=20)

    print("\n== Table 7.12: 1-shot episodes")
    Z = E.compute_all_embeddings(model, g, device)
    labels_all_dev, pools = E.make_pools(g, device)
    results = E.run_episodic_suite(Z, labels_all_dev, pools, episodes, SEED)
    save_json(E.results_to_json(results), os.path.join(paths.episodic_dir, f"episodic_results_{tag}.json"))
    print("META-DDIE paper (accuracy):", E.METADDIE_PAPER)

    print("\n== T3: ChemBERTa-only embeddings")
    t3 = {}
    for mode in ("mean", "hadamard", "set_untrained"):
        Zc = E.build_chemberta_only_embeddings(g, device, fusion_mode=mode, seed=SEED)
        t3.update({(mode,) + k: v for k, v in
                   E.run_episodic_suite(Zc, labels_all_dev, pools, episodes, SEED, seed_salt=mode).items()})
        del Zc
    save_json(E.results_to_json(t3), os.path.join(paths.episodic_dir, f"t3_chemberta_only_{tag}.json"))

    print("\n== Table 7.13 (T4): inference-time smoothing of the frozen embeddings")
    E.set_seed(SEED)
    train_nids = g.ndata["train_mask"].nonzero(as_tuple=False).squeeze().to(device)
    knn_idx, knn_sim = E.build_knn_to_train(Z, train_nids, k=5)
    apply_mask = (((labels_all_dev >= E.FEWER_MIN) & (labels_all_dev <= E.FEWER_MAX))
                  | ((labels_all_dev >= E.RARE_MIN) & (labels_all_dev <= E.RARE_MAX)))
    print(f"smoothing applies to {int(apply_mask.sum())} nodes (fewer + rare)")
    t4 = {}
    for cfg, mode, params in E.T4_FUSION_CONFIGS:
        Zs = E.compute_smoothed_Z(Z, knn_idx, knn_sim, apply_mask, fusion_mode=mode, fusion_params=params)
        t4.update({(cfg,) + k: v for k, v in
                   E.run_episodic_suite(Zs, labels_all_dev, pools, episodes, SEED, seed_salt=cfg).items()})
        del Zs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    save_json(E.results_to_json(t4), os.path.join(paths.episodic_dir, f"t4_smoothing_{tag}.json"))


if __name__ == "__main__":
    main()
