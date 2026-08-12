import os
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


def load_elliptic_csvs(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the Elliptic feature, class, and edge CSVs.

    Returns
    -------
    raw_feat : (N, 165) float32
    label_series : (N,) int, where 1=illicit, 0=licit, -1=unknown
    time_steps : (N,) int
    tx_ids : (N,) original transaction IDs
    edge_pairs : (E, 2) int64, directed edges before symmetrization
    """
    feat_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    class_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
    edge_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")

    for path in (feat_path, class_path, edge_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                "Download the Elliptic dataset from Kaggle and place the "
                "3 CSVs in data/elliptic/"
            )

    # Features: txId, time step, then 165 node features.
    feat_df = pd.read_csv(feat_path, header=None)

    if feat_df.shape[1] != 167:
        raise ValueError(
            "Expected 167 columns in the feature file "
            f"(txId + time step + 165 features), found {feat_df.shape[1]}."
        )

    tx_ids = feat_df.iloc[:, 0].values
    time_steps = feat_df.iloc[:, 1].values.astype(int)
    raw_feat = feat_df.iloc[:, 2:].values.astype(np.float32)

    if len(np.unique(tx_ids)) != len(tx_ids):
        raise ValueError("Duplicate transaction IDs found in the feature file.")

    if not np.all(np.isin(time_steps, np.arange(1, 50))):
        raise ValueError("Expected time steps to be in the range 1..49.")

    # Handle both string and integer representations of the class column.
    class_df = pd.read_csv(class_path)

    if class_df.shape[1] != 2:
        raise ValueError(
            "Expected 2 columns in the class file "
            f"(txId, class), found {class_df.shape[1]}."
        )

    class_df.columns = ["txId", "class"]

    label_map = {
        "1": 1,
        "2": 0,
        1: 1,
        2: 0,
        "unknown": -1,
    }

    class_df["label"] = class_df["class"].map(label_map).fillna(-1).astype(int)

    # Align labels to the feature-file transaction order.
    label_series = (
        pd.Series(tx_ids, name="txId")
        .to_frame()
        .merge(class_df[["txId", "label"]], on="txId", how="left")["label"]
        .fillna(-1)
        .astype(int)
        .values
    )

    edge_df = pd.read_csv(edge_path)

    edge_df.columns = ["src", "dst"]

    edge_df["src"] = edge_df["src"].astype(np.int64)
    edge_df["dst"] = edge_df["dst"].astype(np.int64)

    tx_ids = tx_ids.astype(np.int64)

    tx_id_to_idx = {
        tid: i
        for i, tid in enumerate(tx_ids)
    }

    valid = (
        edge_df["src"].isin(tx_id_to_idx)
        & edge_df["dst"].isin(tx_id_to_idx)
    )

    dropped = int((~valid).sum())
    edge_df = edge_df[valid]

    if dropped:
        print(
            f"Edges   : dropped {dropped:,} edges referencing "
            "transactions absent from the feature file"
        )

    src = edge_df["src"].map(tx_id_to_idx).to_numpy()
    dst = edge_df["dst"].map(tx_id_to_idx).to_numpy()

    edge_pairs = np.stack(
        [src, dst],
        axis=1,
    ).astype(np.int64)

    return raw_feat, label_series, time_steps, tx_ids, edge_pairs


def _normalize_features(
    raw_feat: np.ndarray,
    label_series: np.ndarray,
    time_steps: np.ndarray,
) -> np.ndarray:
    """Fit StandardScaler on labeled training-period nodes and transform all nodes."""
    train_mask = (time_steps <= 30) & (label_series != -1)

    if not np.any(train_mask):
        raise ValueError("No labeled training-period nodes found.")

    scaler = StandardScaler()
    scaler.fit(raw_feat[train_mask])

    return scaler.transform(raw_feat).astype(np.float32)


def _build_sparse_adjacency(
    edge_pairs: np.ndarray,
    num_nodes: int,
) -> Tuple[
    tf.sparse.SparseTensor,
    tf.sparse.SparseTensor,
    np.ndarray,
]:
    """
    Build the graph representations used by GCN, GraphSAGE, and GAT.

    GCN uses D^-1/2 (A + I) D^-1/2.
    GraphSAGE uses D^-1 A without self-loops.
    GAT uses the plain undirected edge index.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")

    if edge_pairs.ndim != 2 or edge_pairs.shape[1] != 2:
        raise ValueError(
            f"edge_pairs must have shape (E, 2), got {edge_pairs.shape}."
        )

    if edge_pairs.size == 0:
        raise ValueError("The graph contains no edges.")

    src, dst = edge_pairs[:, 0], edge_pairs[:, 1]

    # Make the transaction graph undirected.
    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])

    undirected = np.unique(
        np.stack([all_src, all_dst], axis=1),
        axis=0,
    )

    u_src, u_dst = undirected[:, 0], undirected[:, 1]

    # Remove existing self-loops; GCN/GAT add them explicitly below.
    not_self_loop = u_src != u_dst
    u_src = u_src[not_self_loop]
    u_dst = u_dst[not_self_loop]

    # GCN: A_hat = D^-1/2 (A + I) D^-1/2
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

    order = np.lexsort((gcn_indices[:, 1], gcn_indices[:, 0]))
    gcn_indices = gcn_indices[order].astype(np.int64)
    gcn_vals = gcn_vals[order].astype(np.float32)

    gcn_adj = tf.sparse.SparseTensor(
        indices=gcn_indices,
        values=gcn_vals,
        dense_shape=(num_nodes, num_nodes),
    )

    # GraphSAGE: row-normalized neighbor aggregation, no self-loops.
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

    # GAT adds self-loops internally.
    edge_index = np.stack([u_src, u_dst], axis=0).astype(np.int64)

    return gcn_adj, sage_adj, edge_index


class EllipticGraph:
    """Container for node features, labels, graph representations, and masks."""

    def __init__(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        gcn_adj: tf.sparse.SparseTensor,
        sage_adj: tf.sparse.SparseTensor,
        edge_index: np.ndarray,
        train_mask: tf.Tensor,
        val_mask: tf.Tensor,
        test_mask: tf.Tensor,
        time_step: tf.Tensor,
    ):
        self.x = x
        self.y = y
        self.gcn_adj = gcn_adj
        self.sage_adj = sage_adj
        self.edge_index = edge_index
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.time_step = time_step

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
         return int(self.edge_index.shape[1])
    @property
    def gat_edge_index(self):
        return self.edge_index

def build_graph(data_dir: str) -> EllipticGraph:
    """
    Load the dataset, normalize features, build graph representations, and
    create temporal train/validation/test masks.

    The project uses transductive temporal node classification: the full graph
    and node features are available for message passing, while labels are split
    into train (1-30), validation (31-34), and test (35-49).
    """
    raw_feat, label_series, time_steps, tx_ids, edge_pairs = load_elliptic_csvs(
        data_dir
    )

    features = _normalize_features(
        raw_feat,
        label_series,
        time_steps,
    )

    num_nodes = features.shape[0]

    gcn_adj, sage_adj, edge_index = _build_sparse_adjacency(
        edge_pairs,
        num_nodes,
    )

    labeled = label_series != -1
    train_mask = labeled & (time_steps <= 30)
    val_mask = labeled & (time_steps >= 31) & (time_steps <= 34)
    test_mask = labeled & (time_steps >= 35)

    if not np.any(train_mask):
        raise ValueError("Training split contains no labeled nodes.")

    if not np.any(val_mask):
        raise ValueError("Validation split contains no labeled nodes.")

    if not np.any(test_mask):
        raise ValueError("Test split contains no labeled nodes.")

    graph = EllipticGraph(
        x=tf.constant(features, dtype=tf.float32),
        y=tf.constant(label_series, dtype=tf.int32),
        gcn_adj=gcn_adj,
        sage_adj=sage_adj,
        edge_index=edge_index,
        train_mask=tf.constant(train_mask, dtype=tf.bool),
        val_mask=tf.constant(val_mask, dtype=tf.bool),
        test_mask=tf.constant(test_mask, dtype=tf.bool),
        time_step=tf.constant(time_steps, dtype=tf.int32),
    )

    print(
        f"Graph  : {graph.num_nodes:,} nodes | "
        f"{graph.num_edges:,} undirected edges | "
        f"{features.shape[1]} features"
    )
    print(
        f"Train  : {int(train_mask.sum()):,} nodes  "
        f"(illicit={int((label_series[train_mask] == 1).sum()):,})"
    )
    print(
        f"Val    : {int(val_mask.sum()):,} nodes  "
        f"(illicit={int((label_series[val_mask] == 1).sum()):,})"
    )
    print(
        f"Test   : {int(test_mask.sum()):,} nodes  "
        f"(illicit={int((label_series[test_mask] == 1).sum()):,})"
    )

    return graph
