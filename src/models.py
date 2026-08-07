"""
src/models.py

Hand-rolled TensorFlow GNN layers (no Spektral, no PyTorch anywhere) + the 3
tf.keras.Model wrappers (GCN / GraphSAGE / GAT) + a thin Random Forest builder
for symmetry with the GNN builders.

Architecture is fixed (matches the original README's 2-layer design) — hidden
dim, dropout, and attention heads are constructor args, NOT hardcoded, so the
notebook cell that instantiates these controls tuning by passing args at call time.

Conv layers
-----------
GCNConv  : H' = A_hat @ (H @ W) + b, where A_hat = D^-1/2 (A+I) D^-1/2 is the
           precomputed sparse tensor from src/data.py (`graph.gcn_adj`).
SAGEConv : mean-aggregate neighbor features via the precomputed row-normalized
           sparse tensor (`graph.sage_adj`, no self-loops), concat with the
           node's own features, then a single dense layer — matches
           Hamilton et al.'s h_v = sigma(W . CONCAT(h_v, MEAN(neighbors))).
GATConv  : per-edge attention computed directly from `graph.edge_index` (2, E),
           normalized per destination node via segment-softmax
           (tf.math.unsorted_segment_max/_sum), multi-head via stacked output.
           Stays edge-indexed/sparse throughout — never materializes an NxN
           dense attention matrix (~200k nodes in the full Elliptic graph would
           blow up memory otherwise).
"""

import tensorflow as tf


# ---------------------------------------------------------------------------
# GCNConv
# ---------------------------------------------------------------------------
class GCNConv(tf.keras.layers.Layer):
    """H' = A_hat @ (H @ W) + b.  `adj` must be the precomputed D^-1/2(A+I)D^-1/2
    tf.sparse.SparseTensor from src/data.py (graph.gcn_adj)."""

    def __init__(self, out_channels: int, use_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.use_bias = use_bias

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        self.w = self.add_weight(
            name="w", shape=(in_channels, self.out_channels),
            initializer="glorot_uniform", trainable=True,
        )
        if self.use_bias:
            self.b = self.add_weight(
                name="b", shape=(self.out_channels,),
                initializer="zeros", trainable=True,
            )
        super().build(input_shape)

    def call(self, x, adj: tf.sparse.SparseTensor):
        h = tf.matmul(x, self.w)                       # (N, out_channels)
        h = tf.sparse.sparse_dense_matmul(adj, h)       # (N, out_channels)
        if self.use_bias:
            h = h + self.b
        return h


# ---------------------------------------------------------------------------
# SAGEConv
# ---------------------------------------------------------------------------
class SAGEConv(tf.keras.layers.Layer):
    """h_v = W . CONCAT(h_v, MEAN(neighbors)).  `adj` must be the precomputed
    D^-1 A (no self-loops) tf.sparse.SparseTensor from src/data.py (graph.sage_adj)."""

    def __init__(self, out_channels: int, use_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.use_bias = use_bias

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        # weight applies to CONCAT(self, mean_neighbor) -> 2*in_channels
        self.w = self.add_weight(
            name="w", shape=(2 * in_channels, self.out_channels),
            initializer="glorot_uniform", trainable=True,
        )
        if self.use_bias:
            self.b = self.add_weight(
                name="b", shape=(self.out_channels,),
                initializer="zeros", trainable=True,
            )
        super().build(input_shape)

    def call(self, x, adj: tf.sparse.SparseTensor):
        neigh_mean = tf.sparse.sparse_dense_matmul(adj, x)   # (N, in_channels)
        h = tf.concat([x, neigh_mean], axis=-1)              # (N, 2*in_channels)
        h = tf.matmul(h, self.w)
        if self.use_bias:
            h = h + self.b
        return h


# ---------------------------------------------------------------------------
# GATConv
# ---------------------------------------------------------------------------
def _segment_softmax(logits: tf.Tensor, segment_ids: tf.Tensor, num_segments: int):
    """Softmax of `logits` (E, heads) within each group defined by `segment_ids`
    (E,), per head. Numerically stabilized by subtracting the per-segment max."""
    max_per_segment = tf.math.unsorted_segment_max(logits, segment_ids, num_segments)
    logits_shifted = logits - tf.gather(max_per_segment, segment_ids)
    exp_logits = tf.exp(logits_shifted)
    sum_per_segment = tf.math.unsorted_segment_sum(exp_logits, segment_ids, num_segments)
    denom = tf.gather(sum_per_segment, segment_ids)
    return exp_logits / (denom + 1e-16)


class GATConv(tf.keras.layers.Layer):
    """Multi-head graph attention. `edge_index` is the plain (2, E) undirected
    edge array from src/data.py (graph.edge_index); edge_index[0]=source,
    edge_index[1]=destination. Self-loops are added internally so every node
    attends to itself as well as its neighbors (matches PyG's GATConv default)."""

    def __init__(self, out_channels: int, heads: int = 1, concat: bool = True,
                 dropout: float = 0.0, leaky_relu_slope: float = 0.2,
                 use_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout_rate = dropout
        self.leaky_relu_slope = leaky_relu_slope
        self.use_bias = use_bias

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        self.w = self.add_weight(
            name="w", shape=(in_channels, self.heads * self.out_channels),
            initializer="glorot_uniform", trainable=True,
        )
        # attention params split into "target" and "source" halves so
        # a^T [Wh_i || Wh_j] = a_dst . Wh_i + a_src . Wh_j  (avoids ever
        # materializing the concatenated (E, heads, 2*out_channels) tensor)
        self.a_dst = self.add_weight(
            name="a_dst", shape=(self.heads, self.out_channels),
            initializer="glorot_uniform", trainable=True,
        )
        self.a_src = self.add_weight(
            name="a_src", shape=(self.heads, self.out_channels),
            initializer="glorot_uniform", trainable=True,
        )
        out_dim = self.heads * self.out_channels if self.concat else self.out_channels
        if self.use_bias:
            self.b = self.add_weight(
                name="b", shape=(out_dim,), initializer="zeros", trainable=True,
            )
        self.attn_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, edge_index: tf.Tensor, training: bool = False):
        num_nodes = tf.shape(x)[0]

        # add self-loops so every node attends to itself too
        self_loops = tf.stack([tf.range(num_nodes), tf.range(num_nodes)], axis=0)
        self_loops = tf.cast(self_loops, edge_index.dtype)
        full_edge_index = tf.concat([edge_index, self_loops], axis=1)
        src, dst = full_edge_index[0], full_edge_index[1]

        Wh = tf.matmul(x, self.w)                                  # (N, heads*out_ch)
        Wh = tf.reshape(Wh, (num_nodes, self.heads, self.out_channels))

        Wh_src = tf.gather(Wh, src)   # (E, heads, out_ch)  -- neighbor / message source
        Wh_dst = tf.gather(Wh, dst)   # (E, heads, out_ch)  -- attending / target node

        # e_ij = LeakyReLU( a_dst . Wh_i  +  a_src . Wh_j ), i=target(dst), j=source(src)
        e_dst = tf.reduce_sum(Wh_dst * self.a_dst, axis=-1)   # (E, heads)
        e_src = tf.reduce_sum(Wh_src * self.a_src, axis=-1)   # (E, heads)
        e = tf.nn.leaky_relu(e_dst + e_src, alpha=self.leaky_relu_slope)

        alpha = _segment_softmax(e, dst, num_nodes)            # (E, heads), sums to 1 per dst
        alpha = self.attn_dropout(alpha, training=training)

        weighted = Wh_src * alpha[:, :, None]                  # (E, heads, out_ch)
        out = tf.math.unsorted_segment_sum(weighted, dst, num_nodes)  # (N, heads, out_ch)

        if self.concat:
            out = tf.reshape(out, (num_nodes, self.heads * self.out_channels))
        else:
            out = tf.reduce_mean(out, axis=1)                  # (N, out_ch)

        if self.use_bias:
            out = out + self.b
        return out


# ---------------------------------------------------------------------------
# Model wrappers (2-layer architecture, matches original README)
# ---------------------------------------------------------------------------
class GCN(tf.keras.Model):
    """GCNConv(in->hidden) -> ReLU -> Dropout -> GCNConv(hidden->hidden) -> ReLU
    -> Dropout -> Linear(hidden->2)."""

    def __init__(self, hidden: int = 64, out_channels: int = 2, dropout: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GCNConv(hidden)
        self.conv2 = GCNConv(hidden)
        self.linear = tf.keras.layers.Dense(out_channels)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, adj, training: bool = False):
        x = self.conv1(x, adj)
        x = tf.nn.relu(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x, adj)
        x = tf.nn.relu(x)
        x = self.dropout2(x, training=training)
        return self.linear(x)


class GraphSAGE(tf.keras.Model):
    """SAGEConv(in->hidden) -> ReLU -> Dropout -> SAGEConv(hidden->hidden) -> ReLU
    -> Dropout -> Linear(hidden->2)."""

    def __init__(self, hidden: int = 64, out_channels: int = 2, dropout: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = SAGEConv(hidden)
        self.conv2 = SAGEConv(hidden)
        self.linear = tf.keras.layers.Dense(out_channels)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, adj, training: bool = False):
        x = self.conv1(x, adj)
        x = tf.nn.relu(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x, adj)
        x = tf.nn.relu(x)
        x = self.dropout2(x, training=training)
        return self.linear(x)


class GAT(tf.keras.Model):
    """Dropout(x) -> GATConv(in->hidden, heads=4, concat=True) -> ELU -> Dropout
    -> GATConv(hidden*4->hidden, heads=1, concat=False) -> ELU -> Linear(hidden->2)."""

    def __init__(self, hidden: int = 64, out_channels: int = 2, heads: int = 4,
                 dropout: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = GATConv(hidden, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden, heads=1, concat=False, dropout=dropout)
        self.linear = tf.keras.layers.Dense(out_channels)
        self.dropout0 = tf.keras.layers.Dropout(dropout)
        self.dropout1 = tf.keras.layers.Dropout(dropout)

    def call(self, x, edge_index, training: bool = False):
        x = self.dropout0(x, training=training)
        x = self.conv1(x, edge_index, training=training)
        x = tf.nn.elu(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x, edge_index, training=training)
        x = tf.nn.elu(x)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Random Forest baseline builder (thin wrapper, for symmetry with the GNN builders)
# ---------------------------------------------------------------------------
def build_random_forest(n_estimators: int = 300, max_depth=None, random_state: int = 42):
    """Tabular baseline — no graph structure, node features only."""
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
    )
