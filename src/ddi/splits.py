import numpy as np
import torch
import dgl
from sklearn.model_selection import train_test_split


def filter_and_split_graph(
    line_subgraph,
    label_key='label',
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    min_class_samples=4,
    random_state=42,
):
    """
    Filters and stratified-splits a DGL line graph for node classification.

    - Removes unlabeled nodes (label < 0)
    - Removes classes with fewer than `min_class_samples` samples
    - Produces boolean train/val/test masks for node-level tasks
    - Stratified across valid classes

    NOTE: test_size=(1 - train_ratio) evaluates to 0.30000000000000004, not 0.30.
    Re-implementations that typed 0.30 got a split one node off (1,010,793 vs
    1,010,792 training nodes). Always call this function; never retype it.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    labels = line_subgraph.ndata[label_key]
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)

    # 1. unlabeled nodes
    labeled_mask = labels >= 0
    labels_labeled = labels[labeled_mask]
    labels_np = labels_labeled.cpu().numpy()
    if labels_np.size == 0:
        raise RuntimeError("No labeled nodes found (labels >= 0).")

    # 2. classes with >= min_class_samples
    counts = np.bincount(labels_np, minlength=int(labels_np.max()) + 1)
    keep_classes = np.where(counts >= min_class_samples)[0]
    if len(keep_classes) == 0:
        raise RuntimeError(f"No classes have >={min_class_samples} samples.")

    keep_classes_t = torch.as_tensor(keep_classes, dtype=torch.long, device=labels.device)
    keep_mask = labeled_mask & torch.isin(labels, keep_classes_t)
    nodes_to_keep = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

    g_filt = dgl.node_subgraph(line_subgraph, nodes_to_keep)
    labels_f = g_filt.ndata[label_key]
    labels_np_f = labels_f.cpu().numpy()
    Nf = g_filt.num_nodes()

    print(f"Filtered graph: {Nf} nodes | Classes kept: {len(keep_classes)} "
          f"({len(np.unique(labels_np_f))} actually present)")

    # 3. stratified train / val / test
    all_idx_f = np.arange(Nf)
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        all_idx_f, labels_np_f,
        test_size=(1 - train_ratio),
        stratify=labels_np_f,
        random_state=random_state,
        shuffle=True
    )

    # drop temp classes too small to stratify (appear only once)
    temp_classes, temp_counts = np.unique(y_temp, return_counts=True)
    too_small_temp = temp_classes[temp_counts < 2]
    if len(too_small_temp) > 0:
        print(f"Dropping {len(too_small_temp)} class(es) too rare in val/test split: "
              f"{too_small_temp.tolist()}")
        mask_keep_temp = np.isin(y_temp, too_small_temp, invert=True)
        idx_temp = idx_temp[mask_keep_temp]
        y_temp = y_temp[mask_keep_temp]

    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp,
        test_size=(1 - relative_val_ratio),
        stratify=y_temp,
        random_state=random_state,
        shuffle=True
    )

    # 4. boolean masks
    train_mask = torch.zeros(Nf, dtype=torch.bool)
    val_mask = torch.zeros(Nf, dtype=torch.bool)
    test_mask = torch.zeros(Nf, dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    g_filt.ndata['train_mask'] = train_mask
    g_filt.ndata['val_mask'] = val_mask
    g_filt.ndata['test_mask'] = test_mask

    print(f"Split summary: Train={train_mask.sum().item()} | "
          f"Val={val_mask.sum().item()} | Test={test_mask.sum().item()}")
    return g_filt


MIN_CLASS_SAMPLES_EPISODIC = 2
COMMON_MIN, COMMON_MAX = 1, 175
FEWER_MIN, FEWER_MAX = 176, 204
RARE_MIN, RARE_MAX = 205, 227


def build_metaddie_graph(line_graph, seed=42, min_class_samples=MIN_CLASS_SAMPLES_EPISODIC):
    """
    (a) Keep classes with >= 2 samples so rare (205-227) survives.
    (b) Pretraining is on COMMON (labels 1..175) only: train/val/test masks are
        restricted to common. Labels 176..227 and label 0 stay in the graph as
        neighbours but are never trained on.
    (c) Single stratified 70/15/15 split over common nodes.
    """
    labels_all = line_graph.ndata['label'].long()
    num_nodes = line_graph.num_nodes()
    N = labels_all.numpy()

    unique_labels, counts = np.unique(N, return_counts=True)
    eligible_labels = set(unique_labels[counts >= min_class_samples].tolist())
    print(f"Classes with >= {min_class_samples} samples: {len(eligible_labels)}")

    keep_mask = np.array([lbl in eligible_labels for lbl in N], dtype=bool)
    keep_nids = np.where(keep_mask)[0]
    g = dgl.node_subgraph(line_graph, torch.tensor(keep_nids, dtype=torch.long))
    print(f"Graph after min_class_samples filter: {g.num_nodes()} nodes "
          f"(dropped {num_nodes - g.num_nodes()})")

    labels = g.ndata['label'].long().numpy()
    common_mask = (labels >= COMMON_MIN) & (labels <= COMMON_MAX)
    common_nids = np.where(common_mask)[0]
    common_labels = labels[common_mask]
    print(f"Common nodes: {len(common_nids)}")
    print(f"Fewer nodes ({FEWER_MIN}-{FEWER_MAX}): {((labels >= FEWER_MIN) & (labels <= FEWER_MAX)).sum()}")
    print(f"Rare nodes ({RARE_MIN}-{RARE_MAX}):  {((labels >= RARE_MIN) & (labels <= RARE_MAX)).sum()}")

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        common_nids, common_labels, test_size=0.30,
        stratify=common_labels, random_state=seed, shuffle=True,
    )
    uniq_tmp, cnt_tmp = np.unique(y_temp, return_counts=True)
    singletons_in_temp = set(uniq_tmp[cnt_tmp < 2].tolist())
    if singletons_in_temp:
        keep_temp = np.array([y not in singletons_in_temp for y in y_temp])
        print(f"  Note: {len(singletons_in_temp)} class(es) too small to stratify "
              f"val/test; moving their {(~keep_temp).sum()} sample(s) back to train.")
        extra_train = idx_temp[~keep_temp]
        idx_temp = idx_temp[keep_temp]
        y_temp = y_temp[keep_temp]
        idx_train = np.concatenate([idx_train, extra_train])

    idx_val, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=0.50,
        stratify=y_temp, random_state=seed, shuffle=True,
    )

    train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    val_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    test_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask

    print(f"Split: train={train_mask.sum().item()} | "
          f"val={val_mask.sum().item()} | test={test_mask.sum().item()}")
    assert (g.ndata['label'][train_mask].min() >= COMMON_MIN and
            g.ndata['label'][train_mask].max() <= COMMON_MAX), \
        "train_mask leaked outside common range"
    return g


def split_info(g, **extra):
    info = {k: g.ndata[f"{k}_mask"].nonzero(as_tuple=False).squeeze(1).cpu()
            for k in ("train", "val", "test")}
    info["num_nodes"] = g.num_nodes()
    info.update(extra)
    return info


def save_split(g, path, **extra):
    torch.save(split_info(g, **extra), path)
    print(f"saved split -> {path}")


def check_split(g, path):
    """Assert that g carries exactly the split stored at `path`."""
    ref = torch.load(path)
    assert ref["num_nodes"] == g.num_nodes(), "node count differs from saved split"
    for k in ("train", "val", "test"):
        cur = g.ndata[f"{k}_mask"].nonzero(as_tuple=False).squeeze(1).cpu()
        assert torch.equal(cur, ref[k]), f"{k} split differs from {path}"
    print(f"split matches {path}")
