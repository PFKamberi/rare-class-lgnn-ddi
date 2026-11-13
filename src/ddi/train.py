import os
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from dgl.dataloading import DataLoader, NeighborSampler
from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm

from .losses import FocalLoss, LDAMLoss, class_weights_from_train
from .metrics import compute_metrics_default, tail_head_f1_both_versions
from .model import GraphSAGEBlock
from .retrieval import build_retrieval_x_base


@torch.no_grad()
def evaluate_minibatch_with_confidence(model, dataloader, device, label_key='label', logit_scale=1.0):
    """Default-convention metrics + mean max-softmax confidence (logit_scale=s for LDAM)."""
    model.eval()
    preds_all, labels_all, conf_all = [], [], []
    for _, _, blocks in dataloader:
        logits = model(blocks)
        y = blocks[-1].dstdata[label_key].to(device)
        probs = torch.softmax(logit_scale * logits, dim=1)
        conf = probs.max(dim=1).values
        preds = logits.argmax(dim=1)
        preds_all.append(preds.cpu())
        labels_all.append(y.cpu())
        conf_all.append(conf.cpu())
    if len(preds_all) == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "weighted_f1": 0.0, "avg_conf": 0.0}
    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)
    conf_all = torch.cat(conf_all)
    out = compute_metrics_default(labels_all.numpy(), preds_all.numpy())
    out["avg_conf"] = conf_all.mean().item()
    return out


@torch.no_grad()
def collect_preds_labels(model, dataloader, device, label_key='label'):
    """Full y_true / y_pred arrays, in loader order."""
    model.eval()
    preds_all, labels_all = [], []
    for _, _, blocks in dataloader:
        logits = model(blocks)
        y = blocks[-1].dstdata[label_key]
        preds_all.append(logits.argmax(dim=1).cpu())
        labels_all.append(y.cpu())
    if len(preds_all) == 0:
        return None, None
    return torch.cat(labels_all).numpy(), torch.cat(preds_all).numpy()


@torch.no_grad()
def collect_logits_labels(model, dataloader, label_key='label'):
    """Same as collect_preds_labels but keeps the logits (used by diagnostics)."""
    model.eval()
    ys, zs = [], []
    for _, _, blocks in dataloader:
        zs.append(model(blocks).cpu())
        ys.append(blocks[-1].dstdata[label_key].cpu())
    return torch.cat(ys).numpy(), torch.cat(zs).numpy()


def train_model_transductive(
    model,
    g,
    optimizer,
    scheduler=None,
    device=None,
    epochs=100,
    fanouts=[15, 10],
    batch_size=1024,
    num_workers=2,
    feat_key='smiles_embedding',
    label_key='label',
    weight_smoothing=0.05,
    alpha_normalize=False,
    loss_type="cross_entropy",    # "cross_entropy", "focal", "ldam"
    focal_gamma=2.0,
    label_smoothing=None,
    sampler_type="uniform",       # "uniform", "weighted", "per_class_quota"
    max_grad_norm=None,
    early_stopping_patience=6,
    early_stopping_min_delta=5e-4,
    checkpoint_path="best_model.pt",
    checkpoint_dir=None,
    log_every=1,
    tail_threshold=15,
    eval_train_every=1,
    resume_from=None,
    strict_resume=False,
):
    """Train with minibatch neighbour sampling and early stopping on validation
    macro-F1; returns the test metrics of the best checkpoint.

    The thesis settings are passed by run_variant (LDAM loss, label smoothing
    0.05, grad-norm clip 1.0, uniform sampler, num_workers=0). The defaults
    below are the notebook's generic defaults; no thesis run uses them.
    """
    device = device or next(model.parameters()).device

    # ---- checkpoint path ----
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, os.path.basename(checkpoint_path))
    else:
        parent = os.path.dirname(checkpoint_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # ---- masks & node ids ----
    train_mask = g.ndata["train_mask"].bool()
    val_mask = g.ndata["val_mask"].bool()
    test_mask = g.ndata["test_mask"].bool()
    train_nids = train_mask.nonzero(as_tuple=False).squeeze()
    val_nids = val_mask.nonzero(as_tuple=False).squeeze()
    test_nids = test_mask.nonzero(as_tuple=False).squeeze()

    labels = g.ndata[label_key]
    num_classes = model.classifier.out_features

    # ---- class stats from train only ----
    train_labels = labels[train_nids]
    cls_counts = torch.bincount(train_labels, minlength=num_classes).float()
    tail_classes = (cls_counts < tail_threshold)
    class_weights = class_weights_from_train(
        labels, train_mask, num_classes, device,
        smoothing=weight_smoothing, alpha_normalize=alpha_normalize)

    ldam_loss = None
    ldam_s = 30.0
    if loss_type == "ldam":
        ldam_loss = LDAMLoss(cls_num_list=cls_counts.tolist(), max_m=0.5, s=30.0).to(device)
        ldam_s = float(ldam_loss.s)
    eval_logit_scale = ldam_s if loss_type == "ldam" else 1.0

    neighbor_sampler = NeighborSampler(fanouts)
    val_loader = DataLoader(g, val_nids, neighbor_sampler,
                            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(g, test_nids, neighbor_sampler,
                             batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_eval_loader = DataLoader(g, train_nids, neighbor_sampler,
                                   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Training on {len(train_nids)} nodes | Val={len(val_nids)} | Test={len(test_nids)}")
    print(f"Sampler: {sampler_type} | Loss: {loss_type} | Tail<thr={tail_threshold}: "
          f"{tail_classes.sum().item()} classes")
    print(f"Checkpointing: save best only -> {checkpoint_path} | "
          f"early_stopping_patience={early_stopping_patience} | min_delta={early_stopping_min_delta}")

    best_val_f1 = float("-inf")
    best_epoch = 0
    epochs_no_improve = 0
    start_epoch = 1
    loaded_resume_path = None

    # ---- optional resume ----
    if resume_from is not None:
        print(f"Loading checkpoint from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            loaded_resume_path = resume_from
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if scheduler is not None and "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = int(ckpt.get("epoch", 0)) + 1
                best_val_f1 = float(ckpt.get("best_val_f1", float("-inf")))
                best_epoch = int(ckpt.get("best_epoch", max(start_epoch - 1, 0)))
                epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
                print(f"Full resume loaded | start_epoch={start_epoch} | "
                      f"best_val_f1={best_val_f1:.4f} | best_epoch={best_epoch}")
            else:
                msg = "Checkpoint has model_state_dict but no optimizer_state_dict; warm-start only."
                if strict_resume:
                    raise ValueError(msg)
                print(msg)
        else:
            try:
                model.load_state_dict(ckpt)
                loaded_resume_path = resume_from
                msg = "Loaded raw state_dict checkpoint; warm-start, not a full resume."
                if strict_resume:
                    raise ValueError(msg)
                print(msg)
            except Exception as e:
                raise ValueError(f"Could not interpret checkpoint at {resume_from}.\n{e}")

    for epoch in range(start_epoch, epochs + 1):
        # ---- train nodes for this epoch ----
        if sampler_type == "uniform":
            epoch_train_nids = train_nids
        elif sampler_type == "weighted":
            nonzero = cls_counts > 0
            cls_w = torch.zeros_like(cls_counts)
            cls_w[nonzero] = 1.0 / cls_counts[nonzero]
            sample_weights = cls_w[train_labels]
            sampler = WeightedRandomSampler(weights=sample_weights,
                                            num_samples=len(train_nids), replacement=True)
            sampled_idx = torch.tensor(list(sampler), dtype=torch.long)
            epoch_train_nids = train_nids[sampled_idx]
        elif sampler_type == "per_class_quota":
            epoch_size = len(train_nids)
            quota = math.ceil(epoch_size / num_classes)
            per_class_nodes = []
            for c in range(num_classes):
                mask_c = (train_labels == c)
                if not mask_c.any():
                    continue
                nodes_c = train_nids[mask_c]
                idx = torch.randint(low=0, high=len(nodes_c), size=(quota,), dtype=torch.long)
                per_class_nodes.append(nodes_c[idx])
            if len(per_class_nodes) == 0:
                epoch_train_nids = train_nids
            else:
                epoch_train_nids = torch.cat(per_class_nodes)
                perm = torch.randperm(len(epoch_train_nids))
                epoch_train_nids = epoch_train_nids[perm[:epoch_size]]
        else:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")

        train_loader = DataLoader(g, epoch_train_nids, neighbor_sampler,
                                  batch_size=batch_size, shuffle=(sampler_type == "uniform"),
                                  num_workers=num_workers)

        # ---- train epoch ----
        model.train()
        total_loss = total_acc = total_f1 = 0.0
        n_batches = 0
        for _, _, blocks in tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False):
            y = blocks[-1].dstdata[label_key].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(blocks)
            if loss_type == "focal":
                loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)(logits, y)
            elif loss_type == "ldam":
                loss = ldam_loss(logits, y)
            else:
                loss = F.cross_entropy(logits, y, weight=class_weights,
                                       label_smoothing=(label_smoothing or 0.0))
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()
            macro_f1_batch = f1_score(y.detach().cpu(), preds.detach().cpu(),
                                      average="macro", zero_division=0)
            total_loss += loss.item()
            total_acc += acc
            total_f1 += macro_f1_batch
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        # ---- train(eval) + val + tail/head + confidence ----
        if epoch % eval_train_every == 0:
            train_eval_metrics = evaluate_minibatch_with_confidence(
                model, train_eval_loader, device, label_key=label_key, logit_scale=eval_logit_scale)
            val_metrics = evaluate_minibatch_with_confidence(
                model, val_loader, device, label_key=label_key, logit_scale=eval_logit_scale)
            ytr, ptr = collect_preds_labels(model, train_eval_loader, device, label_key=label_key)
            yva, pva = collect_preds_labels(model, val_loader, device, label_key=label_key)
            (train_tail_strict, train_head_strict, train_tail_present, train_head_present) = \
                tail_head_f1_both_versions(ytr, ptr, tail_classes, num_classes)
            (val_tail_strict, val_head_strict, val_tail_present, val_head_present) = \
                tail_head_f1_both_versions(yva, pva, tail_classes, num_classes)
        else:
            train_eval_metrics = {"acc": float("nan"), "macro_f1": float("nan"), "avg_conf": float("nan")}
            train_tail_strict = train_tail_present = float("nan")
            val_metrics = evaluate_minibatch_with_confidence(
                model, val_loader, device, label_key=label_key, logit_scale=eval_logit_scale)
            val_tail_strict = val_tail_present = float("nan")
            val_head_strict = val_head_present = float("nan")

        val_f1 = val_metrics["macro_f1"]
        improved = val_f1 > (best_val_f1 + early_stopping_min_delta)
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            ckpt_to_save = {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_f1": best_val_f1,
                "epochs_no_improve": epochs_no_improve,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            if scheduler is not None:
                ckpt_to_save["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(ckpt_to_save, checkpoint_path)
        else:
            epochs_no_improve += 1

        if epoch % log_every == 0:
            save_flag = "saved" if improved else "-"
            print(
                f"train(EVAL-epoch{epoch}) acc={train_eval_metrics['acc']:.4f} "
                f"macroF1={train_eval_metrics['macro_f1']:.4f} conf={train_eval_metrics['avg_conf']:.3f} | "
                f"train_tail(strict)={train_tail_strict:.4f} train_tail(present)={train_tail_present:.4f} | "
                f"val acc={val_metrics['acc']:.4f} macroF1={val_f1:.4f} conf={val_metrics['avg_conf']:.3f} | "
                f"val_tail(strict)={val_tail_strict:.4f} val_tail(present)={val_tail_present:.4f} | "
                f"best={best_val_f1:.4f}@{best_epoch} | no_improve={epochs_no_improve}/"
                f"{early_stopping_patience} {save_flag}")

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (best epoch={best_epoch}, "
                  f"best val macroF1={best_val_f1:.4f}).")
            break

    # ---- load best & test ----
    load_path = None
    if os.path.exists(checkpoint_path):
        load_path = checkpoint_path
    elif loaded_resume_path is not None and os.path.exists(loaded_resume_path):
        load_path = loaded_resume_path
    if load_path is not None:
        loaded = torch.load(load_path, map_location=device)
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            model.load_state_dict(loaded["model_state_dict"])
            best_val_f1 = float(loaded.get("best_val_f1", best_val_f1))
            best_epoch = int(loaded.get("best_epoch", best_epoch))
        else:
            model.load_state_dict(loaded)
        print(f"Loaded best model from {load_path} (Val macroF1={best_val_f1:.4f}, epoch={best_epoch})")

    test_metrics = evaluate_minibatch_with_confidence(
        model, test_loader, device, label_key=label_key, logit_scale=eval_logit_scale)
    yte, pte = collect_preds_labels(model, test_loader, device, label_key=label_key)
    (test_tail_strict, test_head_strict, test_tail_present, test_head_present) = \
        tail_head_f1_both_versions(yte, pte, tail_classes, num_classes)

    test_metrics["best_val_macro_f1"] = float(best_val_f1)
    test_metrics["best_epoch"] = int(best_epoch)
    test_metrics["checkpoint_path"] = checkpoint_path

    print("=== Final Test (sklearn-default macro-F1; NOT the reporting convention) ===")
    print(f"Acc={test_metrics['acc']:.4f} | MacroF1={test_metrics['macro_f1']:.4f} | "
          f"MicroF1={test_metrics['micro_f1']:.4f} | Conf={test_metrics['avg_conf']:.3f}")
    print(f"TailF1(strict)={test_tail_strict:.4f} | TailF1(present)={test_tail_present:.4f} | "
          f"HeadF1(strict)={test_head_strict:.4f}")
    return test_metrics


def run_variant(g, run_name, proto_k, proto_tau, proto_weighting, rare_threshold, proto_fusion_mode,
                proto_apply, fusion_kwargs, num_epochs, batch_size,
                dropout=0.1, fusion_type="set", fusion_heads=4, run_baseline=False,
                checkpoint_dir=None, knn_cache_path=None, device=None,
                early_stopping_patience=6, early_stopping_min_delta=5e-4,
                resume_from=None, strict_resume=False):
    """One warm-start configuration, exactly as in the thesis notebook."""
    assert checkpoint_dir is not None, "pass checkpoint_dir (see ddi.paths)"
    print("=" * 90)
    print(f"Running variant: {run_name}")
    print("=" * 90)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_feats = g.ndata['drug_u_emb'].shape[1]
    num_classes = int(g.ndata['label'].max().item() + 1)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if run_baseline:
        model = GraphSAGEBlock(
            in_feats=in_feats, hidden_feats=num_classes, num_classes=num_classes,
            dropout=dropout, fusion_type=fusion_type, fusion_heads=fusion_heads,
            proto_module=None,
        ).to(device)
    else:
        proto_module, rare_node_mask = build_retrieval_x_base(
            g, num_classes, proto_k=proto_k, proto_tau=proto_tau,
            proto_weighting=proto_weighting, rare_threshold=rare_threshold,
            device=device, knn_cache_path=knn_cache_path)
        model = GraphSAGEBlock(
            in_feats=in_feats, hidden_feats=num_classes, num_classes=num_classes,
            dropout=dropout, fusion_type=fusion_type, fusion_heads=fusion_heads,
            proto_module=proto_module, proto_fusion_mode=proto_fusion_mode,
            proto_apply=proto_apply, rare_node_mask=rare_node_mask,
            proto_fusion_kwargs=fusion_kwargs,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    test_metrics = train_model_transductive(
        model, g, optimizer=optimizer, scheduler=scheduler, device=device,
        epochs=num_epochs, fanouts=[15, 10], batch_size=batch_size, num_workers=0,
        loss_type="ldam", sampler_type="uniform", label_smoothing=0.05,
        weight_smoothing=0.05, alpha_normalize=False, max_grad_norm=1.0,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        checkpoint_path=f"ckpt_{run_name}.pt", checkpoint_dir=checkpoint_dir,
        log_every=1, resume_from=resume_from, strict_resume=strict_resume,
    )

    final_save_path = os.path.join(checkpoint_dir, f"final_model_{run_name}.pt")
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")
    print(f"\nDone: {run_name}")
    print(test_metrics)
    return model, test_metrics


# The five Chapter 7 configurations (THIS_IS_THE_ONE_I_RUN cells 24-33).
# The concat run was resumed once in the thesis (cell 32 -> 33); a fresh run
# does not need that.
WARM_START_VARIANTS = {
    "baseline": dict(proto_k=None, proto_tau=None, proto_weighting=None, rare_threshold=15,
                     proto_fusion_mode=None, proto_apply=None, fusion_kwargs=None,
                     run_baseline=True, early_stopping_patience=6),
    "proto_alpha_rare_only_k5": dict(proto_k=5, proto_tau=0.07, proto_weighting="softmax",
                                     rare_threshold=15, proto_fusion_mode="alpha",
                                     proto_apply="rare_only", fusion_kwargs=None,
                                     early_stopping_patience=10),
    "proto_alpha_rare_only_k3": dict(proto_k=3, proto_tau=0.07, proto_weighting="softmax",
                                     rare_threshold=15, proto_fusion_mode="alpha",
                                     proto_apply="rare_only", fusion_kwargs=None,
                                     early_stopping_patience=10),
    "proto_gate_rare_only_k5": dict(proto_k=5, proto_tau=0.07, proto_weighting="softmax",
                                    rare_threshold=15, proto_fusion_mode="gate",
                                    proto_apply="rare_only", fusion_kwargs=None,
                                    early_stopping_patience=10),
    "proto_concat_rare_only_k5": dict(proto_k=5, proto_tau=0.07, proto_weighting="softmax",
                                      rare_threshold=15, proto_fusion_mode="concat",
                                      proto_apply="rare_only", fusion_kwargs=None,
                                      early_stopping_patience=10),
}
COMMON_RUN_ARGS = dict(num_epochs=100, batch_size=512, dropout=0.1, fusion_type="set", fusion_heads=4)
