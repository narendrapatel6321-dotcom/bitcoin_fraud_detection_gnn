"""
Evaluation, visualization, and reproducibility utilities.
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def set_seed(seed: int = 42):
    """Set random seeds for Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Metrics

def _metrics(
    true: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict:
    """Compute classification metrics for the illicit class."""
    if len(np.unique(true)) > 1:
        roc_auc = roc_auc_score(true, probs)
        pr_auc = average_precision_score(true, probs)
    else:
        roc_auc = 0.0
        pr_auc = 0.0

    return {
        "f1_illicit": f1_score(
            true,
            preds,
            pos_label=1,
            zero_division=0,
        ),
        "f1_macro": f1_score(
            true,
            preds,
            average="macro",
            zero_division=0,
        ),
        "precision": precision_score(
            true,
            preds,
            pos_label=1,
            zero_division=0,
        ),
        "recall": recall_score(
            true,
            preds,
            pos_label=1,
            zero_division=0,
        ),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold": threshold,
        "probs": probs,
        "preds": preds,
        "true": true,
    }


def select_threshold(
    true: np.ndarray,
    probs: np.ndarray,
) -> float:
    """
    Select the probability threshold that maximizes illicit-class F1.

    This must be applied to validation predictions only. The selected threshold
    should then be frozen before evaluating the test set.
    """
    if len(np.unique(true)) < 2:
        raise ValueError(
            "Threshold selection requires both classes in the validation set."
        )

    thresholds = np.unique(probs)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        preds = (probs >= threshold).astype(np.int32)

        f1 = f1_score(
            true,
            preds,
            pos_label=1,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold


def evaluate_gnn(
    model: tf.keras.Model,
    graph,
    mask: tf.Tensor,
    inputs,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate a trained GNN on the nodes selected by `mask`.

    The model performs a full-graph forward pass; metrics are computed only on
    the supplied mask.
    """
    logits = model(
        graph.x,
        inputs,
        training=False,
    )

    masked_logits = tf.boolean_mask(
        logits,
        mask,
    )

    probs = tf.nn.softmax(
        masked_logits,
        axis=1,
    )[:, 1].numpy()

    preds = (probs >= threshold).astype(np.int32)

    true = tf.boolean_mask(
        graph.y,
        mask,
    ).numpy()

    return _metrics(
        true,
        preds,
        probs,
        threshold,
    )


def evaluate_sklearn(
    model,
    X,
    y,
    threshold: float = 0.5,
) -> dict:
    """Evaluate a trained sklearn classifier using the supplied threshold."""
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(np.int32)

    return _metrics(
        np.asarray(y),
        preds,
        probs,
        threshold,
    )


def print_report(name: str, metrics: dict):
    """Print a concise evaluation report."""
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Illicit F1 : {metrics['f1_illicit']:.4f}")
    print(f"  Macro F1   : {metrics['f1_macro']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  PR-AUC     : {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"  Threshold  : {metrics['threshold']:.4f}")

    cm = confusion_matrix(
        metrics["true"],
        metrics["preds"],
    )

    print(f"  Confusion matrix:\n{cm}")

# Plots

def plot_training_history(
    histories: dict,
    save_dir: str = "results",
):
    """Plot training loss and validation illicit F1."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    for name, history in histories.items():
        if "train_loss" in history:
            axes[0].plot(
                history["train_loss"],
                label=name,
            )

        if "val_f1" in history:
            axes[1].plot(
                history["val_f1"],
                label=name,
            )

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].set_title("Validation Illicit F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score (illicit class)")
    axes[1].legend()

    plt.tight_layout()

    path = os.path.join(
        save_dir,
        "training_history.png",
    )

    plt.savefig(
        path,
        dpi=150,
    )
    print(f"Saved: {path}")
    plt.close()


def plot_confusion_matrices(
    results: dict,
    save_dir: str = "results",
):
    """Plot confusion matrices for the supplied model results."""
    os.makedirs(save_dir, exist_ok=True)

    n = len(results)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5 * n, 4),
    )

    if n == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(
        axes,
        results.items(),
    ):
        cm = confusion_matrix(
            metrics["true"],
            metrics["preds"],
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Licit", "Illicit"],
            yticklabels=["Licit", "Illicit"],
        )

        ax.set_title(
            f"{name}\nF1={metrics['f1_illicit']:.3f}"
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()

    path = os.path.join(
        save_dir,
        "confusion_matrices.png",
    )

    plt.savefig(
        path,
        dpi=150,
    )
    print(f"Saved: {path}")
    plt.close()


def plot_comparison_bar(
    results: dict,
    save_dir: str = "results",
):
    """Plot the main test-set metrics for model comparison."""
    os.makedirs(save_dir, exist_ok=True)

    metrics = [
        "f1_illicit",
        "precision",
        "recall",
        "pr_auc",
        "roc_auc",
    ]

    labels = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(
        figsize=(12, 6),
    )

    for i, name in enumerate(labels):
        values = [
            results[name][metric]
            for metric in metrics
        ]

        ax.bar(
            x + i * width,
            values,
            width,
            label=name,
        )

    ax.set_xticks(
        x + width * (len(labels) - 1) / 2
    )

    ax.set_xticklabels(
        [
            "Illicit F1",
            "Precision",
            "Recall",
            "PR-AUC",
            "ROC-AUC",
        ]
    )

    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison on Test Set")
    ax.legend()

    plt.tight_layout()

    path = os.path.join(
        save_dir,
        "model_comparison.png",
    )

    plt.savefig(
        path,
        dpi=150,
    )
    print(f"Saved: {path}")
    plt.close()

# Model saving

def save_model(
    model: tf.keras.Model,
    name: str,
    save_dir: str = "results",
):
    """Save a trained Keras model's weights."""
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(
        save_dir,
        f"{name}.weights.h5",
    )

    model.save_weights(path)

    print(f"Saved model: {path}")
