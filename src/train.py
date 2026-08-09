"""
Training utilities for the GNN models and Random Forest baseline.

Contains class weighting, the GNN training loop with validation-based early
stopping, and Random Forest training.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tqdm import tqdm

from .models import build_random_forest
from .data import EllipticGraph


def compute_class_weights(graph: EllipticGraph) -> tf.Tensor:
    """Compute inverse-frequency class weights from labeled training nodes."""
    y_train = tf.boolean_mask(graph.y, graph.train_mask).numpy()

    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    weights = 1.0 / (counts + 1e-6)
    weights /= weights.sum()

    return tf.constant(weights, dtype=tf.float32)


def _weighted_sparse_ce(
    labels: tf.Tensor,
    logits: tf.Tensor,
    class_weights: tf.Tensor,
) -> tf.Tensor:
    """
    Compute a weighted mean sparse cross-entropy loss.

    Normalizing by the sum of sample weights preserves the intended weighted
    mean rather than taking a plain mean after applying the weights.
    """
    sample_weights = tf.gather(class_weights, labels)

    per_sample_loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels,
        logits,
        from_logits=True,
    )

    return tf.reduce_sum(
        per_sample_loss * sample_weights
    ) / tf.reduce_sum(sample_weights)


def _val_illicit_f1(
    model: tf.keras.Model,
    graph: EllipticGraph,
    inputs,
    y_val_np: np.ndarray,
) -> float:
    """Compute illicit-class F1 on the validation nodes for early stopping."""
    logits = model(graph.x, inputs, training=False)

    val_logits = tf.boolean_mask(
        logits,
        graph.val_mask,
    )

    val_preds = tf.argmax(
        val_logits,
        axis=1,
    ).numpy()

    return f1_score(
        y_val_np,
        val_preds,
        pos_label=1,
        zero_division=0,
    )


def train_gnn(
    model: tf.keras.Model,
    graph: EllipticGraph,
    inputs,
    epochs: int = 200,
    lr: float = 0.01,
    patience: int = 20,
    weight_decay: float = 5e-4,
    verbose: bool = True,
):
    """
    Train a GNN on the full transductive graph.

    The loss is computed only on training nodes. Validation illicit F1 is used
    for early stopping, and the best validation weights are restored.
    """
    if isinstance(inputs, np.ndarray):
        inputs = tf.constant(inputs)

    class_weights = compute_class_weights(graph)

    # Keras applies weight decay directly to the model weights.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        weight_decay=weight_decay,
    )

    y_train = tf.boolean_mask(
        graph.y,
        graph.train_mask,
    )

    y_val_np = tf.boolean_mask(
        graph.y,
        graph.val_mask,
    ).numpy()

    best_val_f1 = -1.0
    best_weights = None
    no_improve = 0

    history = {
        "train_loss": [],
        "val_f1": [],
    }

    epoch_iter = (
        tqdm(
            range(1, epochs + 1),
            desc="Training",
            unit="ep",
        )
        if verbose
        else range(1, epochs + 1)
    )

    for epoch in epoch_iter:
        with tf.GradientTape() as tape:
            # Full-graph forward pass; loss is masked to training nodes.
            logits = model(
                graph.x,
                inputs,
                training=True,
            )

            train_logits = tf.boolean_mask(
                logits,
                graph.train_mask,
            )

            loss = _weighted_sparse_ce(
                y_train,
                train_logits,
                class_weights,
            )

        grads = tape.gradient(
            loss,
            model.trainable_variables,
        )

        optimizer.apply_gradients(
            zip(grads, model.trainable_variables)
        )

        val_f1 = _val_illicit_f1(
            model,
            graph,
            inputs,
            y_val_np,
        )

        history["train_loss"].append(
            float(loss.numpy())
        )
        history["val_f1"].append(
            float(val_f1)
        )

        if verbose:
            epoch_iter.set_postfix(
                loss=f"{loss.numpy():.4f}",
                val_f1=f"{val_f1:.4f}",
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = [
                weight.numpy().copy()
                for weight in model.trainable_variables
            ]
            no_improve = 0

        else:
            no_improve += 1

            if no_improve >= patience:
                if verbose:
                    print(
                        f"\nEarly stop at epoch {epoch} "
                        f"(best val F1={best_val_f1:.4f})"
                    )
                break

    if best_weights is not None:
        for variable, best_value in zip(
            model.trainable_variables,
            best_weights,
        ):
            variable.assign(best_value)

    return model, history


def train_rf(
    graph: EllipticGraph,
    n_estimators: int = 300,
    max_depth=None,
    random_state: int = 42,
):
    """Train the graph-free Random Forest baseline on training nodes."""
    X_train = graph.x.numpy()[graph.train_mask.numpy()]
    y_train = graph.y.numpy()[graph.train_mask.numpy()]

    # Class imbalance is handled by class_weight="balanced" in the RF builder.
    clf = build_random_forest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    clf.fit(
        X_train,
        y_train,
    )

    return clf
