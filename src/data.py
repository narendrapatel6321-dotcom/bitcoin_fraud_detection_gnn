"""
src/data.py

TensorFlow port of the Elliptic Bitcoin dataset loader + graph builder.

Merges the old `data_loader.py` + `graph_builder.py` into a single module:
  - CSV ingestion (features, classes, edgelist)
  - Label encoding (1 -> illicit(1), 2 -> licit(0), unknown -> -1)
  - StandardScaler normalization, fit ONLY on labeled train-period nodes (ts <= 30)
    to prevent leakage from val/test periods
  - Undirected edge construction
  - Self-loop addition + symmetric-normalized sparse adjacency
    (tf.sparse.SparseTensor) for GCNConv, precomputed once here
  - Row-normalized (mean-aggregator) sparse adjacency for SAGEConv, precomputed once here
  - Plain undirected edge_index (2, E) for GATConv, which computes its own
    per-edge attention + segment-softmax rather than using a precomputed matrix
  - Temporal train/val/test boolean masks

No tuning knobs live here — this is stable, fixed logic (per the project's
notebook-vs-.py split: anything with hyperparameters goes in a notebook cell,
anything that's fixed mechanics goes in a .py file).

Bug fixed vs. original PyTorch version:
  - PyG's GCNConv adds self-loops (A + I) internally by default. Since GCNConv is
    being hand-rolled here, that step must be done EXPLICITLY when building the
    sparse adjacency below — it is not automatic in plain TensorFlow.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def load_elliptic_csvs(data_dir: str):
    """
    Read the 3 raw Elliptic CSVs and return numpy arrays.

    Returns
    -------
    raw_feat     : (N, 165) float32 — unnormalized node features (txId, time-step cols dropped)
    label_series : (N,) int         — 1=illicit, 0=licit, -1=unknown
    time_steps   : (N,) int         — 1..49
    tx_ids       : (N,)             — original transaction IDs
    edge_pairs   : (E, 2) int64     — directed src/dst node indices (pre-symmetrization)
    """
    feat_path  = os.path.join(data_dir, "elliptic_txs_features.csv")
    class_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
    edge_path  = os.path.join(data_dir, "elliptic_txs_edgelist.csv")

    for p in [feat_path, class_path, edge_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing: {p}\n"
                "Download the Elliptic dataset from Kaggle and place the 3 CSVs in data/elliptic/"
            )

    # --- features (no header): col0=txId, col1=time step, cols2..166=features ---
    feat_df = pd.read_csv(feat_path, header=None)
    tx_ids     = feat_df.iloc[:, 0].values
    time_steps = feat_df.iloc[:, 1].values.astype(int)
    raw_feat   = feat_df.iloc[:, 2:].values.astype(np.float32)  # (N, 165)

    # --- labels ---
    class_df = pd.read_csv(class_path)
    class_df.columns = ["txId", "class"]
    # NOTE: the Kaggle CSV's class column loads as str ("1"/"2"/"unknown") or as int
    # (1/2) depending on pandas dtype inference on a given machine — handle both
    # explicitly rather than assuming one or the other.
    label_map = {"1": 1, "2": 0, 1: 1, 2: 0, "unknown": -1}
    class_df["label"] = class_df["class"].map(label_map).fillna(-1).astype(int)

    label_series = (
        pd.Series(tx_ids, name="txId")
        .to_frame()
        .merge(class_df[["txId", "label"]], on="txId", how="left")["label"]
        .fillna(-1)
        .astype(int)
        .values
    )

    # --- edges (kept directed here; symmetrization happens in _build_sparse_adjacency) ---
    edge_df = pd.read_csv(edge_path, header=None, names=["src", "dst"])
    tx_id_to_idx = {tid: i for i, tid in enumerate(tx_ids)}

    valid = edge_df["src"].isin(tx_id_to_idx) & edge_df["dst"].isin(tx_id_to_idx)
    edge_df = edge_df[valid]

    src = edge_df["src"].map(tx_id_to_idx).values
    dst = edge_df["dst"].map(tx_id_to_idx).values
    edge_pairs = np.stack([src, dst], axis=1).astype(np.int64)  # (E, 2)

    return raw_feat, label_series, time_steps, tx_ids, edge_pairs


def _normalize_features(raw_feat: np.ndarray, label_series: np.ndarray,
                         time_steps: np.ndarray) -> np.ndarray:
    """Fit StandardScaler on labeled train-period nodes only (ts<=30), apply to all nodes."""
    train_mask = (time_steps <= 30) & (label_series != -1)
    scaler = StandardScaler()
    scaler.fit(raw_feat[train_mask])
    return scaler.transform(raw_feat).astype(np.float32)


def _build_sparse_adjacency(edge_pairs: np.ndarray, num_nodes: int):
    """
    From directed edge pairs, build everything the 3 hand-rolled GNN layers need:

      gcn_adj    : tf.sparse.SparseTensor (N, N), A_hat = D^-1/2 (A + I) D^-1/2
                   (symmetric-normalized, self-loops added explicitly)
      sage_adj   : tf.sparse.SparseTensor (N, N), A_mean = D^-1 A
                   (row-normalized mean aggregator, NO self-loops — SAGEConv
                   concatenates the node's own features with the aggregated
                   neighbor features separately, so self-loops here would double-count)
      edge_index : (2, E) int64 np.ndarray, plain undirected edge list for GATConv,
                   which computes per-edge attention scores directly and normalizes
                   them via segment-softmax rather than using a precomputed matrix
    """
    src, dst = edge_pairs[:, 0], edge_pairs[:, 1]

    # symmetrize: make every directed edge bidirectional, then dedupe
    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])
    undirected = np.unique(np.stack([all_src, all_dst], axis=1), axis=0)
    u_src, u_dst = undirected[:, 0], undirected[:, 1]

    # Defensively drop any self-loops already present in the raw edge list
    # (a transaction referencing itself would be unusual but not impossible in
    # real data). GCN and GAT both add exactly one self-loop per node explicitly
    # below; if the raw data already contained one, that node would end up with
    # TWO (i, i) entries at the same sparse-tensor coordinate. tf.sparse ops don't
    # guarantee duplicate coordinates get summed the way the math intends, so this
    # keeps every node's self-loop unique instead of relying on that assumption.
    not_self_loop = u_src != u_dst
    u_src, u_dst = u_src[not_self_loop], u_dst[not_self_loop]

    # ---- GCN adjacency: A_hat = D^-1/2 (A + I) D^-1/2 ----
    self_loops = np.arange(num_nodes)
    gcn_src = np.concatenate([u_src, self_loops])
    gcn_dst = np.concatenate([u_dst, self_loops])

    deg = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(deg, gcn_src, 1.0)
    deg_inv_sqrt = np.zeros(num_nodes, dtype=np.float32)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)

    gcn_vals = deg_inv_sqrt[gcn_src] * deg_inv_sqrt[gcn_dst]
    gcn_indices = np.stack([gcn_src, gcn_dst], axis=1)

    # tf.sparse.SparseTensor wants row-major sorted indices
    order = np.lexsort((gcn_indices[:, 1], gcn_indices[:, 0]))
    gcn_indices = gcn_indices[order].astype(np.int64)
    gcn_vals = gcn_vals[order].astype(np.float32)

    gcn_adj = tf.sparse.SparseTensor(
        indices=gcn_indices,
        values=gcn_vals,
        dense_shape=(num_nodes, num_nodes),
    )

    # ---- SAGE mean-aggregator adjacency: A_mean = D^-1 A  (no self-loops) ----
    deg_no_loop = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(deg_no_loop, u_src, 1.0)
    deg_inv = np.zeros(num_nodes, dtype=np.float32)
    nonzero2 = deg_no_loop > 0
    deg_inv[nonzero2] = 1.0 / deg_no_loop[nonzero2]

    sage_vals = deg_inv[u_src]
    sage_indices = np.stack([u_src, u_dst], axis=1)
    order2 = np.lexsort((sage_indices[:, 1], sage_indices[:, 0]))
    sage_indices = sage_indices[order2].astype(np.int64)
    sage_vals = sage_vals[order2].astype(np.float32)

    sage_adj = tf.sparse.SparseTensor(
        indices=sage_indices,
        values=sage_vals,
        dense_shape=(num_nodes, num_nodes),
    )

    # plain undirected edge_index for GAT (no self-loops added here — GATConv will
    # add its own self-attention term explicitly, same reasoning as PyG's GATConv)
    edge_index = np.stack([u_src, u_dst], axis=0).astype(np.int64)

    return gcn_adj, sage_adj, edge_index


class EllipticGraph:
    """Plain container mirroring the old torch_geometric.data.Data object, TF-native."""

    def __init__(self, x, y, gcn_adj, sage_adj, edge_index,
                 train_mask, val_mask, test_mask, time_step):
        self.x           = x            # (N, 165) float32 tf.Tensor
        self.y           = y            # (N,) int32 tf.Tensor, values in {-1, 0, 1}
        self.gcn_adj     = gcn_adj      # tf.sparse.SparseTensor, D^-1/2 (A+I) D^-1/2
        self.sage_adj    = sage_adj     # tf.sparse.SparseTensor, D^-1 A (mean aggregator)
        self.edge_index  = edge_index   # (2, E) int64 np.ndarray, undirected, for GAT
        self.train_mask  = train_mask   # (N,) bool tf.Tensor
        self.val_mask    = val_mask     # (N,) bool tf.Tensor
        self.test_mask   = test_mask    # (N,) bool tf.Tensor
        self.time_step   = time_step    # (N,) int32 tf.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


def build_graph(data_dir: str) -> EllipticGraph:
    """Load CSVs, normalize features, build TF sparse adjacencies + temporal masks.

    Train mask : time steps 1-30,  labeled (label != -1)
    Val mask   : time steps 31-34, labeled
    Test mask  : time steps 35-49, labeled
    """
    raw_feat, label_series, time_steps, tx_ids, edge_pairs = load_elliptic_csvs(data_dir)
    features = _normalize_features(raw_feat, label_series, time_steps)

    num_nodes = features.shape[0]
    gcn_adj, sage_adj, edge_index = _build_sparse_adjacency(edge_pairs, num_nodes)

    labeled    = label_series != -1
    train_mask = labeled & (time_steps <= 30)
    val_mask   = labeled & (time_steps >= 31) & (time_steps <= 34)
    test_mask  = labeled & (time_steps >= 35)

    graph = EllipticGraph(
        x           = tf.constant(features, dtype=tf.float32),
        y           = tf.constant(label_series, dtype=tf.int32),
        gcn_adj     = gcn_adj,
        sage_adj    = sage_adj,
        edge_index  = edge_index,
        train_mask  = tf.constant(train_mask, dtype=tf.bool),
        val_mask    = tf.constant(val_mask, dtype=tf.bool),
        test_mask   = tf.constant(test_mask, dtype=tf.bool),
        time_step   = tf.constant(time_steps, dtype=tf.int32),
    )

    print(f"Graph  : {graph.num_nodes:,} nodes | {graph.num_edges:,} undirected edges | {features.shape[1]} features")
    print(f"Train  : {int(train_mask.sum()):,} nodes  (illicit={int((label_series[train_mask] == 1).sum()):,})")
    print(f"Val    : {int(val_mask.sum()):,} nodes  (illicit={int((label_series[val_mask] == 1).sum()):,})")
    print(f"Test   : {int(test_mask.sum()):,} nodes  (illicit={int((label_series[test_mask] == 1).sum()):,})")

    return graph
