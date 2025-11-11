import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
import numpy as np
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder

def _as_rank3(t):
    if t is None:
        return None
    t = tf.convert_to_tensor(t)
    if t.shape.rank == 2:
        return t[..., tf.newaxis]
    return t

class MLP(Layer):
    def __init__(self, units, activations, use_bias):
        super().__init__()
        self._layers = []
        for i, u in enumerate(units):
            self._layers.append(Dense(u, activation=activations[i], use_bias=use_bias))
    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

class UpdateEmbeddings(Layer):
    def __init__(self, msg_dims, hidden_units, mlp_layers, from_to_ind,
                 reduce_op="sum", activation="relu", use_bias=True,
                 use_residual=True, context_dims=0):
        super().__init__()
        from_to_ind = np.asarray(from_to_ind, dtype=np.int32)
        self._from = tf.constant(from_to_ind[:, 0], dtype=tf.int32)
        self._to   = tf.constant(from_to_ind[:, 1], dtype=tf.int32)

        self._reduce_op = reduce_op
        self._residual = use_residual
        self._ctx_dims = context_dims

        self._msg_mlp = MLP([hidden_units]*(mlp_layers-1) + [msg_dims],
                            [activation]*(mlp_layers-1) + [None],
                            use_bias)

        self._upd_mlp = MLP([hidden_units]*(mlp_layers-1) + [msg_dims],
                            [activation]*(mlp_layers-1) + [None],
                            use_bias)

    def call(self, h_from, h_to, context=None):
        g_from = tf.gather(h_from, self._from, axis=1)
        g_to   = tf.gather(h_to,   self._to,   axis=1)
        msgs = self._msg_mlp(tf.concat([g_from, g_to], axis=-1))

        B = tf.shape(h_to)[0]
        N = tf.shape(h_to)[1]
        E = tf.shape(msgs)[1]
        D = tf.shape(msgs)[-1]

        msgs_flat = tf.reshape(msgs, [B*E, D])
        idx = tf.reshape(self._to[None,:] + (tf.range(B) * N)[:,None], [-1])

        if self._reduce_op == "sum":
            agg = tf.math.unsorted_segment_sum(msgs_flat, idx, B*N)
        else:
            agg = tf.math.unsorted_segment_mean(msgs_flat, idx, B*N)

        agg = tf.reshape(agg, [B, N, D])
        inp = [agg, h_to]
        if context is not None and self._ctx_dims>0:
            inp.append(context)

        delta = self._upd_mlp(tf.concat(inp, axis=-1))
        return h_to + delta if self._residual else delta

class GNN_BP(Layer):
    def __init__(self, pcm, num_iter=10, num_embed_dims=16, num_msg_dims=16,
                 num_hidden_units=48, num_mlp_layers=3, reduce_op="sum",
                 activation="relu", use_bias=True, output_all_iter=False, clip_llr_to=None,
                 use_attributes=False, node_attribute_dims=0, msg_attribute_dims=0):
        super().__init__()
        pcm = np.asarray(pcm, dtype=np.int32)
        self._nc, self._nv = pcm.shape
        self._num_iter = num_iter
        self._output_all_iter = output_all_iter

        edges = np.stack(np.where(pcm), axis=1).astype(np.int32)

        self._init = Dense(num_embed_dims)
        self._read = Dense(1)

        self._cn_upd = UpdateEmbeddings(num_msg_dims, num_hidden_units, num_mlp_layers,
                                        np.flip(edges,1), context_dims=0,
                                        reduce_op=reduce_op, activation=activation,
                                        use_bias=use_bias)

        self._vn_upd = UpdateEmbeddings(num_msg_dims, num_hidden_units, num_mlp_layers,
                                        edges, context_dims=num_embed_dims,
                                        reduce_op=reduce_op, activation=activation,
                                        use_bias=use_bias)

    def call(self, llr, context):
        llr = _as_rank3(llr)
        context = _as_rank3(context)
        h_init = self._init(tf.concat([llr, context], axis=-1))
        h_vn = h_init
        h_cn = tf.zeros_like(h_vn)

        predictions = []
        for _ in range(self._num_iter):
            h_cn = self._cn_upd(h_vn, h_cn)
            h_vn = self._vn_upd(h_cn, h_vn, context=h_init)
            if self._output_all_iter:
                predictions.append(tf.squeeze(self._read(h_vn), -1))

        if self._output_all_iter:
            return predictions
        else:
            return tf.squeeze(self._read(h_vn), -1)

class LDPC5GGNN(GNN_BP):
    def __init__(self, encoder, num_iter=10, return_infobits=False, **kwargs):
        dec = LDPC5GDecoder(encoder, prune_pcm=True)
        pcm = dec.pcm.toarray()
        super().__init__(pcm, num_iter=num_iter, **kwargs)
        self._k = encoder.k
        self._nv = pcm.shape[1]
        self._ret = return_infobits

    def call(self, inputs):
        llr, context = inputs
        llr = _as_rank3(llr)
        context = _as_rank3(context)

        def pad(t):
            pad_len = self._nv - tf.shape(t)[1]
            return tf.pad(t, [[0,0],[0,pad_len],[0,0]])

        llr = tf.cond(tf.shape(llr)[1]<self._nv, lambda: pad(llr), lambda: llr)
        context = tf.cond(tf.shape(context)[1] < self._nv, lambda: pad(context), lambda: context)

        out = super().call(llr, context)

        if isinstance(out, list):
            if self._ret:
                return [o[:,:self._k] for o in out]
            else:
                return out
        else:
            return out[:,:self._k] if self._ret else out
