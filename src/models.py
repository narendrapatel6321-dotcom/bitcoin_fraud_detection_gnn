"""
Custom TensorFlow GNN layers and model wrappers for the Elliptic dataset.
"""

import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

# GCNConv

class GCNConv(tf.keras.layers.Layer):
    """Graph convolution using a precomputed normalized sparse adjacency."""

    def __init__(
        self,
        out_channels: int,
        use_bias: bool = True,
        **kwargs,
    ):
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.use_bias = use_bias

    def build(self, input_shape):
        in_channels = int(input_shape[-1])

        self.w = self.add_weight(
            name="w",
            shape=(in_channels, self.out_channels),
            initializer="glorot_uniform",
            trainable=True,
        )

        if self.use_bias:
            self.b = self.add_weight(
                name="b",
                shape=(self.out_channels,),
                initializer="zeros",
                trainable=True,
            )

        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        adj: tf.sparse.SparseTensor,
    ) -> tf.Tensor:
        h = tf.matmul(x, self.w)
        h = tf.sparse.sparse_dense_matmul(adj, h)

        if self.use_bias:
            h = h + self.b

        return h

# SAGEConv

class SAGEConv(tf.keras.layers.Layer):
    """
    Mean-aggregation GraphSAGE layer using a precomputed row-normalized
    adjacency without self-loops.
    """

    def __init__(
        self,
        out_channels: int,
        use_bias: bool = True,
        **kwargs,
    ):
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.use_bias = use_bias

    def build(self, input_shape):
        in_channels = int(input_shape[-1])

        # Apply the weight to [self representation | neighbor mean].
        self.w = self.add_weight(
            name="w",
            shape=(2 * in_channels, self.out_channels),
            initializer="glorot_uniform",
            trainable=True,
        )

        if self.use_bias:
            self.b = self.add_weight(
                name="b",
                shape=(self.out_channels,),
                initializer="zeros",
                trainable=True,
            )

        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        adj: tf.sparse.SparseTensor,
    ) -> tf.Tensor:
        neigh_mean = tf.sparse.sparse_dense_matmul(adj, x)
        h = tf.concat([x, neigh_mean], axis=-1)
        h = tf.matmul(h, self.w)

        if self.use_bias:
            h = h + self.b

        return h

# GATConv

def _segment_softmax(
    logits: tf.Tensor,
    segment_ids: tf.Tensor,
    num_segments: int,
) -> tf.Tensor:
    """Apply a numerically stable softmax independently within each segment."""
    max_per_segment = tf.math.unsorted_segment_max(
        logits,
        segment_ids,
        num_segments,
    )

    logits_shifted = (
        logits
        - tf.gather(max_per_segment, segment_ids)
    )

    exp_logits = tf.exp(logits_shifted)

    sum_per_segment = tf.math.unsorted_segment_sum(
        exp_logits,
        segment_ids,
        num_segments,
    )

    denom = tf.gather(
        sum_per_segment,
        segment_ids,
    )

    return exp_logits / (denom + 1e-16)


class GATConv(tf.keras.layers.Layer):
    """
    Multi-head graph attention using a plain (2, E) edge index.

    Self-loops are added internally so each node can attend to itself.
    """

    def __init__(
        self,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        leaky_relu_slope: float = 0.2,
        use_bias: bool = True,
        **kwargs,
    ):
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        if heads <= 0:
            raise ValueError("heads must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        if leaky_relu_slope < 0.0:
            raise ValueError("leaky_relu_slope must be non-negative.")

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
            name="w",
            shape=(in_channels, self.heads * self.out_channels),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Splitting the attention vector avoids materializing
        # [Wh_i || Wh_j] for every edge.
        self.a_dst = self.add_weight(
            name="a_dst",
            shape=(self.heads, self.out_channels),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.a_src = self.add_weight(
            name="a_src",
            shape=(self.heads, self.out_channels),
            initializer="glorot_uniform",
            trainable=True,
        )

        out_dim = (
            self.heads * self.out_channels
            if self.concat
            else self.out_channels
        )

        if self.use_bias:
            self.b = self.add_weight(
                name="b",
                shape=(out_dim,),
                initializer="zeros",
                trainable=True,
            )

        self.attn_dropout = tf.keras.layers.Dropout(
            self.dropout_rate
        )

        super().build(input_shape)

    def call(
        self,
        x: tf.Tensor,
        edge_index: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        num_nodes = tf.shape(x)[0]

        # Add one self-loop per node.
        self_loops = tf.stack(
            [
                tf.range(num_nodes),
                tf.range(num_nodes),
            ],
            axis=0,
        )

        self_loops = tf.cast(
            self_loops,
            edge_index.dtype,
        )

        full_edge_index = tf.concat(
            [edge_index, self_loops],
            axis=1,
        )

        src = full_edge_index[0]
        dst = full_edge_index[1]

        Wh = tf.matmul(x, self.w)

        Wh = tf.reshape(
            Wh,
            (
                num_nodes,
                self.heads,
                self.out_channels,
            ),
        )

        Wh_src = tf.gather(Wh, src)
        Wh_dst = tf.gather(Wh, dst)

        # a^T [Wh_i || Wh_j]
        # = a_dst . Wh_i + a_src . Wh_j
        e_dst = tf.reduce_sum(
            Wh_dst * self.a_dst,
            axis=-1,
        )

        e_src = tf.reduce_sum(
            Wh_src * self.a_src,
            axis=-1,
        )

        e = tf.nn.leaky_relu(
            e_dst + e_src,
            alpha=self.leaky_relu_slope,
        )

        # Normalize incoming attention coefficients per destination node.
        alpha = _segment_softmax(
            e,
            dst,
            num_nodes,
        )

        alpha = self.attn_dropout(
            alpha,
            training=training,
        )

        weighted = Wh_src * alpha[:, :, None]

        out = tf.math.unsorted_segment_sum(
            weighted,
            dst,
            num_nodes,
        )

        if self.concat:
            out = tf.reshape(
                out,
                (
                    num_nodes,
                    self.heads * self.out_channels,
                ),
            )
        else:
            out = tf.reduce_mean(
                out,
                axis=1,
            )

        if self.use_bias:
            out = out + self.b

        return out

# Model wrappers

class GCN(tf.keras.Model):
    """Two-layer GCN with ReLU, dropout, and a linear output layer."""

    def __init__(
        self,
        hidden: int = 64,
        out_channels: int = 2,
        dropout: float = 0.5,
        **kwargs,
    ):
        if hidden <= 0:
            raise ValueError("hidden must be positive.")

        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        super().__init__(**kwargs)

        self.conv1 = GCNConv(hidden)
        self.conv2 = GCNConv(hidden)

        self.linear = tf.keras.layers.Dense(
            out_channels
        )

        self.dropout1 = tf.keras.layers.Dropout(
            dropout
        )
        self.dropout2 = tf.keras.layers.Dropout(
            dropout
        )

    def call(
        self,
        x: tf.Tensor,
        adj: tf.sparse.SparseTensor,
        training: bool = False,
    ) -> tf.Tensor:
        x = self.conv1(x, adj)
        x = tf.nn.relu(x)
        x = self.dropout1(
            x,
            training=training,
        )

        x = self.conv2(x, adj)
        x = tf.nn.relu(x)
        x = self.dropout2(
            x,
            training=training,
        )

        return self.linear(x)


class GraphSAGE(tf.keras.Model):
    """Two-layer mean-aggregation GraphSAGE model."""

    def __init__(
        self,
        hidden: int = 64,
        out_channels: int = 2,
        dropout: float = 0.5,
        **kwargs,
    ):
        if hidden <= 0:
            raise ValueError("hidden must be positive.")

        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        super().__init__(**kwargs)

        self.conv1 = SAGEConv(hidden)
        self.conv2 = SAGEConv(hidden)

        self.linear = tf.keras.layers.Dense(
            out_channels
        )

        self.dropout1 = tf.keras.layers.Dropout(
            dropout
        )
        self.dropout2 = tf.keras.layers.Dropout(
            dropout
        )

    def call(
        self,
        x: tf.Tensor,
        adj: tf.sparse.SparseTensor,
        training: bool = False,
    ) -> tf.Tensor:
        x = self.conv1(x, adj)
        x = tf.nn.relu(x)
        x = self.dropout1(
            x,
            training=training,
        )

        x = self.conv2(x, adj)
        x = tf.nn.relu(x)
        x = self.dropout2(
            x,
            training=training,
        )

        return self.linear(x)


class GAT(tf.keras.Model):
    """Two-layer multi-head GAT model."""

    def __init__(
        self,
        hidden: int = 64,
        out_channels: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
        **kwargs,
    ):
        if hidden <= 0:
            raise ValueError("hidden must be positive.")

        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        if heads <= 0:
            raise ValueError("heads must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        super().__init__(**kwargs)

        self.conv1 = GATConv(
            hidden,
            heads=heads,
            concat=True,
            dropout=dropout,
        )

        self.conv2 = GATConv(
            hidden,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.linear = tf.keras.layers.Dense(
            out_channels
        )

        self.dropout0 = tf.keras.layers.Dropout(
            dropout
        )
        self.dropout1 = tf.keras.layers.Dropout(
            dropout
        )

    def call(
        self,
        x: tf.Tensor,
        edge_index: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        x = self.dropout0(
            x,
            training=training,
        )

        x = self.conv1(
            x,
            edge_index,
            training=training,
        )

        x = tf.nn.elu(x)

        x = self.dropout1(
            x,
            training=training,
        )

        x = self.conv2(
            x,
            edge_index,
            training=training,
        )

        x = tf.nn.elu(x)

        return self.linear(x)

# Random Forest baseline

def build_random_forest(
    n_estimators: int = 300,
    max_depth=None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Build the graph-free Random Forest baseline."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced",
    )
