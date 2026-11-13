"""
Metrics.

Two conventions appear in the thesis code:
  * sklearn DEFAULT macro-F1 (average over classes present in y_true or y_pred).
    Used ONLY for training logs and early stopping in the warm-start and
    episodic notebooks. Never report it.
  * STRICT macro-F1 over an explicit label set (absent classes count as 0).
    Every reported number uses this.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics_default(y_true, y_pred):
    """Training-log metrics (sklearn default macro-F1). Not for reporting."""
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def tail_head_f1_both_versions(y_true_np, y_pred_np, tail_classes_bool, num_classes):
    """Tail/head F1, (1) strict over all classes 0..C-1 and (2) over present classes only."""
    all_labels = np.arange(num_classes)
    per_class = f1_score(y_true_np, y_pred_np, labels=all_labels, average=None, zero_division=0)
    tail_mask = tail_classes_bool.cpu().numpy().astype(bool)
    head_mask = ~tail_mask
    strict_tail = per_class[tail_mask].mean() if tail_mask.any() else 0.0
    strict_head = per_class[head_mask].mean() if head_mask.any() else 0.0
    present_classes = np.unique(y_true_np)
    present_mask = np.zeros(num_classes, dtype=bool)
    present_mask[present_classes] = True
    tail_present_mask = tail_mask & present_mask
    head_present_mask = head_mask & present_mask
    present_tail = per_class[tail_present_mask].mean() if tail_present_mask.any() else 0.0
    present_head = per_class[head_present_mask].mean() if head_present_mask.any() else 0.0
    return float(strict_tail), float(strict_head), float(present_tail), float(present_head)


def strict_metrics(y, p, num_classes, tail, tag=""):
    """Warm-start reporting metrics (Table 7.8 / 7.10 convention).

    evt     macro-F1 over the event classes 1..num_classes-1 (class 0 = sampled
            non-interacting pairs is excluded). THIS is "macro-F1" in Ch7.
    default sklearn default, for the record only
    rare    macro-F1 over the rare classes (tail = bool array over classes)
    rec     rare test interactions classified correctly, out of n
    """
    rare_ids = np.arange(num_classes)[tail]
    m = tail[y]
    return dict(
        tag=tag,
        acc=float((y == p).mean()),
        evt=float(f1_score(y, p, average="macro", labels=np.arange(1, num_classes), zero_division=0)),
        default=float(f1_score(y, p, average="macro", zero_division=0)),
        rare=float(f1_score(y, p, average="macro", labels=rare_ids, zero_division=0)),
        rec=int((p[m] == y[m]).sum()),
        n=int(m.sum()),
        ndistinct=int(len(np.unique(p))),
    )


def full_metrics(y, p, C, tcounts, tail_thresholds=(15, 20)):
    """Cold-start reporting metrics (Ch8 convention: strict over all C classes;
    there is no class 0 in the cold-start label space)."""
    per = f1_score(y, p, labels=np.arange(C), average=None, zero_division=0)
    o = {"n": int(len(y)), "acc": float(accuracy_score(y, p)),
         "macro_f1_strict": float(per.mean()),
         "macro_f1_sklearn_default": float(f1_score(y, p, average="macro", zero_division=0)),
         "n_classes_in_truth": int(len(np.unique(y))),
         "n_classes_predicted": int(len(np.unique(p)))}
    tc = tcounts.numpy() if hasattr(tcounts, "numpy") else np.asarray(tcounts)
    for thr in tail_thresholds:
        t = tc < thr
        o[f"tail_f1_thr{thr}"] = float(per[t].mean()) if t.any() else 0.0
        o[f"head_f1_thr{thr}"] = float(per[~t].mean()) if (~t).any() else 0.0
        o[f"n_tail_classes_thr{thr}"] = int(t.sum())
    return o
