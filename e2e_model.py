
"""
e2e.py
------
End-to-end link simulator for BPSK over AWGN or flat Rayleigh fading,
optionally with a 5G LDPC encoder and a trainable decoder (e.g., GNN or WBP).

API
---
E2EModel(encoder, decoder, k, n, fading=False):
    - encoder: a callable that maps (batch, k) -> (batch, n) bits. If None, "uncoded" mode is used
               and k must equal n.
    - decoder: a callable that maps (llr_code, training=False) -> llr_hat (either n LLRs or k LLRs).
               If None, uncoded detection is used (llr_hat = channel LLRs on the bits).

call(batch_size, ebnodb, training=False) -> (b_info, llr_hat, loss)
    - b_info: ground-truth information bits, shape [B, k]
    - llr_hat: predicted logits/LLRs for the information bits, shape [B, k]
    - loss: scalar Binary Cross-Entropy-with-logits over information bits

Notes
-----
- BPSK mapping: 0 -> +1, 1 -> -1.
- Perfect CSI for Rayleigh: receiver knows h; LLR uses matched filtering.
- BER can be computed using (tf.cast(tf.less(llr_hat, 0.), tf.int32) != b_info).
"""

from typing import Optional, Tuple
import numpy as np
import tensorflow as tf

Tensor = tf.Tensor

def _to_float(x) -> Tensor:
    if isinstance(x, (float, int)):
        return tf.constant(float(x), tf.float32)
    return tf.cast(x, tf.float32)

def ebnodb_to_sigma2(ebnodb: Tensor, rate: Tensor) -> Tensor:
    """Return per-dimension noise variance sigma^2 from Eb/N0 (dB) and code rate R.
    For BPSK with unit symbol energy, Es = 1 and Eb = Es / R, so
        N0 = Eb / (10^(Eb/N0_dB/10)); sigma^2 = N0/2.
    """
    ebnodb = _to_float(ebnodb)
    rate = _to_float(rate)
    ebno_lin = tf.pow(10.0, ebnodb / 10.0)
    eb = 1.0 / tf.maximum(rate, 1e-6)  # Es=1 -> Eb=1/R; avoid div-by-zero for uncoded (R=1)
    n0 = eb / tf.maximum(ebno_lin, 1e-12)
    sigma2 = n0 * 0.5
    return sigma2

def bpsk_mod(bits: Tensor) -> Tensor:
    """0->+1, 1->-1"""
    return 1.0 - 2.0*tf.cast(bits, tf.float32)

def add_awgn(x: Tensor, sigma2: Tensor, rng=None) -> Tensor:
    """Add AWGN with variance sigma2 per real dimension."""
    if rng is None:
        rng = tf.random.get_global_generator()
    noise = tf.sqrt(sigma2) * tf.random.normal(tf.shape(x), dtype=tf.float32)
    return x + noise

def add_rayleigh(x: Tensor, sigma2: Tensor, rng=None) -> Tuple[Tensor, Tensor]:
    """Flat Rayleigh fading with perfect CSI. Returns (y, h). h>=0 (Rayleigh magnitude)."""
    if rng is None:
        rng = tf.random.get_global_generator()
    # Rayleigh magnitude via sigma=1/sqrt(2) for |h| with unit power E[h^2]=1
    h = tf.sqrt(-2.0 * tf.math.log(1.0 - tf.random.uniform(tf.shape(x), dtype=tf.float32)))
    y = h * x + tf.sqrt(sigma2) * tf.random.normal(tf.shape(x), dtype=tf.float32)
    return y, h

def llr_awgn(y: Tensor, sigma2: Tensor) -> Tensor:
    """LLR for BPSK over AWGN with known sigma^2: L = 2*y/sigma^2"""
    return 2.0 * y / tf.maximum(sigma2, 1e-12)

def llr_rayleigh(y: Tensor, h: Tensor, sigma2: Tensor) -> Tensor:
    """Coherent BPSK LLR with known flat fading h: L = 2*h*y/sigma^2"""
    return 2.0 * h * y / tf.maximum(sigma2, 1e-12)

class E2EModel(tf.keras.Model):
    def __init__(self,
                 encoder: Optional[tf.keras.layers.Layer],
                 decoder: Optional[tf.keras.layers.Layer],
                 k: int,
                 n: int,
                 fading: bool = False,
                 modulation: str = "bpsk",
                 num_bits_per_symbol: int = 1,
                 name: str = "E2EModel"):
        super().__init__(name=name)
        if encoder is None and k != n:
            raise ValueError("Uncoded mode requires k == n.")
        self.encoder = encoder
        self.decoder = decoder
        self.k = int(k)
        self.n = int(n)
        self.fading = bool(fading)
        self.modulation = modulation
        self.num_bits_per_symbol = num_bits_per_symbol

    @property
    def code_rate(self) -> Tensor:
        return tf.constant(float(self.k)/float(self.n), tf.float32)

    def gen_bits(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Generate information bits and (if coded) code bits."""
        b_info = tf.random.uniform(shape=(batch_size, self.k), minval=0, maxval=2, dtype=tf.int32)
        if self.encoder is None:
            b_code = b_info
        else:
            # Some encoders expect float inputs, but 5G encoders accept ints; cast if needed.
            b_code = self.encoder(tf.cast(b_info, tf.int32))
        b_code = tf.cast(b_code, tf.int32)
        return b_info, b_code

    def _channel_step(self, s_code: Tensor, ebnodb: Tensor) -> Tuple[Tensor, Tensor]:
        """Transmit symbols s_code through channel and return (codeword LLRs, maybe aux)."""
        R = self.code_rate if self.encoder is not None else tf.constant(1.0, tf.float32)
        sigma2 = ebnodb_to_sigma2(ebnodb, R)

        if not self.fading:
            y = add_awgn(s_code, sigma2)
            llr_cw = llr_awgn(y, sigma2)
            return llr_cw, y
        else:
            y, h = add_rayleigh(s_code, sigma2)
            llr_cw = llr_rayleigh(y, h, sigma2)
            return llr_cw, h

    def _select_info_llr(self, llr_cw: Tensor) -> Tensor:
        """Return LLRs corresponding to information bits (shape [B, k]).
        If decoder already outputs k LLRs, pass-through.
        Else assume systematic code and take first k bits.
        """
        if llr_cw.shape.rank == 2 and llr_cw.shape[-1] == self.k:
            return llr_cw
        return llr_cw[:, :self.k]

    def call(self, batch_size: int, ebnodb: Tensor, training: bool = False,return_llr: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """Simulate one mini-batch.
        Returns: (b_info, llr_hat_info, loss)
        """
        ebnodb = _to_float(ebnodb)

        # ---- Source & encoder ----
        b_info, b_code = self.gen_bits(batch_size)
        s_code = bpsk_mod(b_code)  # shape [B, n]

        # ---- Channel ----
        llr_cw, h = self._channel_step(s_code, ebnodb)
        if return_llr:
            return llr_cw


        # ---- Decoder ----
        if self.decoder is None:
            # Uncoded or identity: use channel LLRs on information bits
            llr_hat_info = self._select_info_llr(llr_cw)
        else:
            # Many decoders accept (llr, training) or just (llr,). Try both.
            try:
                dec_out = self.decoder((llr_cw,h), training=training)
            except TypeError:
                dec_out = self.decoder((llr_cw,h))
            # dec_out can be (logits, aux) or logits; normalize to logits
            if isinstance(dec_out, (tuple, list)):
                dec_logits = dec_out[0]
            else:
                dec_logits = dec_out
            llr_hat_info = self._select_info_llr(dec_logits)

        # ---- Loss on information bits (BCE with logits) ----
        # Cast labels to float in {0,1}
        labels = tf.cast(b_info, tf.float32)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=llr_hat_info))

        return tf.cast(b_info, tf.int32), tf.cast(llr_hat_info, tf.float32), tf.cast(loss, tf.float32)

def hard_decisions(logits: Tensor) -> Tensor:
    """Return hard bits from logits/LLRs: <0 -> 1, >=0 -> 0 (since 0 maps to +1)."""
    return tf.cast(tf.less(logits, 0.0), tf.int32)
