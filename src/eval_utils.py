"""
src/eval_utils.py

Merge of the original `evaluate.py` + `utils.py`:
  - `_metrics()` / `evaluate_gnn()` / `evaluate_sklearn()` / `print_report()` —
    the final TEST-set metrics dict (F1/precision/recall/AUC/confusion matrix),
    computed ONCE at the end of training. (Not to be confused with
    `src/train.py`'s `_val_illicit_f1()`, which is a lightweight internal helper
    used every epoch purely to drive early stopping.)
  - `plot_training_history()` / `plot_confusion_matrices()` / `plot_comparison_bar()`
    — unchanged plotting logic from the original `utils.py`; framework-agnostic,
    they only ever touched plain numpy arrays, so nothing here needed to change
    for the TF port.
  - `save_model()` — TF-native: Keras 3 `model.save_weights(...)` requires the
    `.weights.h5` extension, replacing torch's `torch.save(model.state_dict(), ...)`.
    Only used for the 3 GNN models (matches the original's scope — RF was never
    saved via this function either).
  - `set_seed()` — TF-native: adds `tf.random.set_seed(...)` alongside the
    original's `random`/`numpy` seeding.

All fixed/stable helpers, no tuning knobs — per the project's notebook-vs-.py split.
"""

import os
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metrics(true, preds, probs) -> dict:
    return {
        "f1_illicit":  f1_score(true, preds, pos_label=1, zero_division=0),
        "f1_macro":    f1_score(true, preds, average="macro", zero_division=0),
        "precision":   precision_score(true, preds, pos_label=1, zero_division=0),
        "recall":      recall_score(true, preds, pos_label=1, zero_division=0),
        "auc":         roc_auc_score(true, probs) if len(np.unique(true)) > 1 else 0.0,
        "preds":       preds,
        "true":        true,
    }


def evaluate_gnn(model, graph, mask: tf.Tensor, inputs) -> dict:
    """
    One forward pass on the full graph, metrics computed on `mask` only.

    model  : trained src.models.GCN / GraphSAGE / GAT
    graph  : src.data.EllipticGraph
    mask   : boolean tf.Tensor, e.g. graph.test_mask
    inputs : the adjacency / edge_index this model needs —
             graph.gcn_adj (GCN), graph.sage_adj (GraphSAGE), graph.edge_index (GAT)
    """
    logits = model(graph.x, inputs, training=False)
    masked_logits = tf.boolean_mask(logits, mask)
    probs = tf.nn.softmax(masked_logits, axis=1)[:, 1].numpy()
    preds = tf.argmax(masked_logits, axis=1).numpy()
    true = tf.boolean_mask(graph.y, mask).numpy()
    return _metrics(true, preds, probs)


def evaluate_sklearn(model, X, y) -> dict:
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return _metrics(y, preds, probs)


def print_report(name: str, metrics: dict):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Illicit F1 : {metrics['f1_illicit']:.4f}")
    print(f"  Macro F1   : {metrics['f1_macro']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  AUC-ROC    : {metrics['auc']:.4f}")
    cm = confusion_matrix(metrics["true"], metrics["preds"])
    print(f"  Confusion matrix:\n{cm}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_training_history(histories: dict, save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, h in histories.items():
        if "train_loss" in h:
            axes[0].plot(h["train_loss"], label=name)
        if "val_f1" in h:
            axes[1].plot(h["val_f1"], label=name)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].set_title("Validation Illicit F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score (illicit class)")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_confusion_matrices(results: dict, save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, m) in zip(axes, results.items()):
        cm = confusion_matrix(m["true"], m["preds"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Licit", "Illicit"],
            yticklabels=["Licit", "Illicit"],
        )
        ax.set_title(f"{name}\nF1={m['f1_illicit']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def plot_comparison_bar(results: dict, save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    metrics = ["f1_illicit", "f1_macro", "precision", "recall", "auc"]
    labels  = list(results.keys())
    x       = np.arange(len(metrics))
    width   = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(labels):
        vals = [results[name][m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=name)

    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(["Illicit F1", "Macro F1", "Precision", "Recall", "AUC-ROC"])
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison on Test Set")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Model saving
# ---------------------------------------------------------------------------
def save_model(model, name: str, save_dir: str = "results"):
    """Save a trained Keras GNN model's weights. Model must already have been
    called on data at least once (true after train_gnn) so its variables exist.
    Keras 3 requires the '.weights.h5' extension."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.weights.h5")
    model.save_weights(path)
    print(f"Saved model: {path}")
