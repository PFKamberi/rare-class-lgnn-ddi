"""Command-line options and graph loading shared by the scripts."""
import argparse
import gc
import json
import os
import warnings

import torch

from ddi.paths import get_paths

warnings.filterwarnings("ignore", message="Dataloader CPU affinity opt is not enabled")


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-root", default=os.environ.get("DDI_DATA_ROOT"),
                   help="folder with the thesis inputs, read only (or set DDI_DATA_ROOT)")
    p.add_argument("--work-root", default=os.environ.get("DDI_WORK_ROOT"),
                   help="folder for all outputs (or set DDI_WORK_ROOT)")
    p.add_argument("--test", action="store_true",
                   help="use the small test subgraph built by 01_prepare_data.py --test; "
                        "checks that the code runs, the numbers mean nothing")
    p.add_argument("--n-test-drugs", type=int, default=800,
                   help="drugs in the test subgraph (default 800)")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    return p


def setup(args):
    if not args.data_root or not args.work_root:
        raise SystemExit("set --data-root and --work-root (or DDI_DATA_ROOT and DDI_WORK_ROOT)")
    paths = get_paths(os.path.abspath(os.path.expanduser(args.data_root)),
                      os.path.abspath(os.path.expanduser(args.work_root)))
    paths.make_output_dirs()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tag = f"test{args.n_test_drugs}" if args.test else "full"
    print(f"data root: {paths.data_root}\nwork root: {paths.work_root}\n"
          f"device: {device} | mode: {'TEST' if args.test else 'FULL'} | tag: {tag}")
    return paths, device, tag


def test_pair_path(paths, n):
    return os.path.join(paths.test_graph_dir, f"test_pair_{n}.bin")


def test_drug_graph_path(paths, n):
    return os.path.join(paths.test_graph_dir, f"test_drug_graph_noneg_{n}.bin")


def load_line_graph(paths, args):
    """Thesis [drug graph, line graph] (or the test pair) with endpoint features attached."""
    from ddi import graph as G
    if args.test:
        path = test_pair_path(paths, args.n_test_drugs)
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found: run 01_prepare_data.py --test first")
    else:
        G.ensure_graph_bin(paths.graph_zip, paths.graph_bin)
        path = paths.graph_bin          # 640M line-graph edges: needs about 64 GB of RAM
    sub, lg = G.load_graph_pair(path)
    return G.attach_endpoint_features(sub, lg)


def warm_graph(paths, args, tag):
    """Warm-start graph with the canonical split; the split is saved once and checked after."""
    from ddi.splits import check_split, filter_and_split_graph, save_split
    lg = load_line_graph(paths, args)
    g = filter_and_split_graph(lg, label_key="label")
    del lg
    gc.collect()
    split_path = os.path.join(paths.warm_dir, f"split_{tag}.pt")
    if os.path.exists(split_path):
        check_split(g, split_path)
    else:
        save_split(g, split_path)
    return g, int(g.ndata["label"].max().item() + 1)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"saved {path}")
