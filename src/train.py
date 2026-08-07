"""
src/train.py

Class-weight computation, the GradientTape training loop (mirrors the original
`train.py`'s epoch loop 1:1, just TF-native), early stopping on validation
illicit-F1 with best-checkpoint restore, and the Random Forest `.fit()` wrapper.

Deliberately self-contained: does NOT import from `src/eval_utils.py` (that file
comes after this one in the build order). Early stopping needs a val-F1 number
every epoch, so this module has its own tiny `_val_illicit_f1()` helper for that
one purpose. The full metrics dict / plots / reporting used for final TEST-set
evaluation live in `src/eval_utils.py`, not here.

Mechanics (epoch loop, early stopping, checkpoint restore) are fixed/stable.
Hyperparameter *values* (epochs, lr, patience) are passed in as args, so the
notebook cell that calls `train_gnn(...)` controls tuning at the call site.

Model / adjacency pairing (per src/models.py's call signatures):
  GCN        -> graph.gcn_adj
  GraphSAGE  -> graph.sage_adj
  GAT        -> graph.edge_index  (converted to tf.Tensor if it's still a numpy array)
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

from .models import build_random_forest


def compute_class_weights(graph) -> tf.Tensor:
    """Inverse-frequency class weights from the training-mask labels only.
    Matches torch's `F.cross_entropy(..., weight=w)` convention: index 0 = licit
    weight, index 1 = illicit weight."""
    y_train = tf.boolean_mask(graph.y, graph.train_mask).numpy()
    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    weights = 1.0 / (counts + 1e-6)
    weights /= weights.sum()
    return tf.constant(weights, dtype=tf.float32)


def _weighted_sparse_ce(labels: tf.Tensor, logits: tf.Tensor, class_weights: tf.Tensor) -> tf.Tensor:
    """Weighted mean cross-entropy, matching torch's `weight=` reduction convention:
    sum(loss_i * weight[y_i]) / sum(weight[y_i])  — NOT a plain mean of per-sample
    weighted losses, which would double-count the effect of the weights."""
    sample_weights = tf.gather(class_weights, labels)
    per_sample_loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )
    return tf.reduce_sum(per_sample_loss * sample_weights) / tf.reduce_sum(sample_weights)


def _val_illicit_f1(model, graph, inputs, y_val_np: np.ndarray) -> float:
    """One forward pass + illicit-class F1 on the validation mask. Used only to
    drive early stopping during training — NOT the final reported metric (that's
    `evaluate_gnn` in src/eval_utils.py, computed once on the test set at the end)."""
    logits = model(graph.x, inputs, training=False)
    val_logits = tf.boolean_mask(logits, graph.val_mask)
    val_preds = tf.argmax(val_logits, axis=1).numpy()
    return f1_score(y_val_np, val_preds, pos_label=1, zero_division=0)


def train_gnn(model, graph, inputs, epochs: int = 200, lr: float = 0.01,
              patience: int = 20, weight_decay: float = 5e-4, verbose: bool = True):
    """
    Train a GCN / GraphSAGE / GAT model on the full (transductive) graph.

    Parameters
    ----------
    model   : a src.models.GCN / GraphSAGE / GAT instance
    graph   : src.data.EllipticGraph
    inputs  : the adjacency or edge_index this model needs —
              graph.gcn_adj for GCN, graph.sage_adj for GraphSAGE,
              graph.edge_index for GAT (numpy array is fine, converted below)
    epochs, lr, patience, weight_decay : hyperparameters (tune from the notebook)

    Returns
    -------
    model   : same model instance, weights restored to the best-val-F1 checkpoint
    history : {"train_loss": [...], "val_f1": [...]}  (one entry per epoch run)
    """
    if isinstance(inputs, np.ndarray):
        inputs = tf.constant(inputs)

    class_weights = compute_class_weights(graph)
    # NOTE: tf.keras.optimizers.Adam's weight_decay is decoupled (AdamW-style —
    # applied directly to the weights), whereas PyTorch's torch.optim.Adam applies
    # weight_decay as a coupled L2 term added to the gradient before the Adam
    # update. Same regularization intent as the original, not bit-identical math.
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay)

    y_train = tf.boolean_mask(graph.y, graph.train_mask)
    y_val_np = tf.boolean_mask(graph.y, graph.val_mask).numpy()

    best_val_f1 = -1.0
    best_weights = None
    no_improve = 0
    history = {"train_loss": [], "val_f1": []}

    epoch_iter = tqdm(range(1, epochs + 1), desc="Training", unit="ep") if verbose else range(1, epochs + 1)
    for epoch in epoch_iter:
        with tf.GradientTape() as tape:
            logits = model(graph.x, inputs, training=True)
            train_logits = tf.boolean_mask(logits, graph.train_mask)
            loss = _weighted_sparse_ce(y_train, train_logits, class_weights)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        val_f1 = _val_illicit_f1(model, graph, inputs, y_val_np)

        history["train_loss"].append(float(loss.numpy()))
        history["val_f1"].append(float(val_f1))

        if verbose:
            epoch_iter.set_postfix(loss=f"{loss.numpy():.4f}", val_f1=f"{val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = [w.numpy().copy() for w in model.trainable_variables]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stop at epoch {epoch} (best val F1={best_val_f1:.4f})")
                break

    if best_weights is not None:
        for var, best_val in zip(model.trainable_variables, best_weights):
            var.assign(best_val)

    return model, history


def train_rf(graph):
    """Random Forest baseline — node features only, no graph structure."""
    X_train = graph.x.numpy()[graph.train_mask.numpy()]
    y_train = graph.y.numpy()[graph.train_mask.numpy()]
    sample_weights = compute_sample_weight("balanced", y_train)

    clf = build_random_forest()
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    return clf
