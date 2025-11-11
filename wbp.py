# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

# Sionna imports
from sionna.phy.fec.ldpc.decoding import LDPCBPDecoder
from sionna.phy.utils.misc import ebnodb2no
from sionna.phy.utils.metrics import compute_ber

# Custom or lightweight drop-ins if needed
def hard_decisions(llr):
    return tf.cast(llr < 0, tf.int32)

def bitwise_mutual_information(bits, llr):
    b = tf.cast(bits, tf.float32)
    term = tf.math.log(1.0 + tf.exp(-(1.0 - 2.0 * b) * tf.cast(llr, tf.float32))) / tf.math.log(2.0)
    return 1.0 - tf.reduce_mean(term)

class WeightedBP(tf.keras.Model):
    """
    System model for training Weighted BP decoding on AWGN or Rayleigh fading channels.
    """
    def __init__(self, pcm, num_iter=5, fading=False):
        super().__init__()
        self._fading = fading
        self._num_iter = num_iter
        self._n = pcm.shape[1]
        self._coderate = 1 - pcm.shape[0] / pcm.shape[1]
        self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # The decoder must be trainable=True to learn weights
        self.decoder = LDPCBPDecoder(pcm,
                                     num_iter=1,
                                     stateful=True,
                                     hard_out=False,
                                     cn_type="boxplus",
                                     trainable=True)

    def call(self, batch_size, ebno_db):
        # 1. Setup noise variance based on EbNo
        # For QPSK/BPSK (2 bits/sym for QPSK but effectively 1 bit/dim BPSK here)
        # We use num_bits_per_symbol=2 to match standard QPSK mapping often used.
        # Adjust if you strictly want BPSK (num_bits_per_symbol=1).
        no = ebnodb2no(ebno_db, num_bits_per_symbol=2, coderate=self._coderate)

        # 2. All-zero codeword (c=0 -> x=+1)
        c = tf.zeros([batch_size, self._n], dtype=tf.float32)
        x = tf.ones([batch_size, self._n], dtype=tf.complex64) # BPSK equivalent: 0 -> +1

        # 3. Generate Channel & LLRs
        if self._fading:
            # Rayleigh Fading: h ~ CN(0, 1)
            h_real = tf.random.normal([batch_size, self._n], stddev=tf.sqrt(0.5))
            h_imag = tf.random.normal([batch_size, self._n], stddev=tf.sqrt(0.5))
            h = tf.complex(h_real, h_imag)
        else:
            # AWGN: h = 1
            h = tf.ones([batch_size, self._n], dtype=tf.complex64)

        # Complex noise w ~ CN(0, No)
        n_std = tf.sqrt(no / 2.0)
        w = tf.complex(tf.random.normal([batch_size, self._n], stddev=n_std),
                       tf.random.normal([batch_size, self._n], stddev=n_std))

        # Received signal y = hx + w
        y = h * x + w

        # LLR calculation: LLR = 4 * Re(h' * y) / No
        # (Assuming bit 0 -> +1 mapping)
        llr = 4.0 * tf.math.real(tf.math.conj(h) * y) / no

        # 4. Multi-loss decoding
        loss = 0
        msg_vn = None # Initialize internal decoder state
        for _ in range(self._num_iter):
            # Perform one iteration, return soft-values (c_hat) and new state (msg_vn)
            c_hat, msg_vn = self.decoder((llr, msg_vn))
            loss += self._bce(c, c_hat)

        loss /= self._num_iter # Average loss over all iterations
        return c, c_hat, loss

    def train_wbp(self, train_param):
        print(f"Training WeightedBP (Fading={self._fading}) with params: {train_param}")
        # Basic sanity checks
        assert len(train_param["wbp_batch_size"]) == len(train_param["wbp_train_iter"])
        assert len(train_param["wbp_batch_size"]) == len(train_param["wbp_learning_rate"])
        assert len(train_param["wbp_batch_size"]) == len(train_param["wbp_ebno_train"])

        optimizer = tf.keras.optimizers.Adam(train_param["wbp_learning_rate"][0])
        bmi = BitwiseMutualInformation() if 'BitwiseMutualInformation' in globals() else bitwise_mutual_information

        for idx, batch_size in enumerate(train_param["wbp_batch_size"]):
            optimizer.learning_rate = train_param["wbp_learning_rate"][idx]
            print(f"Phase {idx+1}/{len(train_param['wbp_batch_size'])}: LR={optimizer.learning_rate.numpy():.1e}, BS={batch_size}, EbNo={train_param['wbp_ebno_train'][idx]}")

            for it in range(train_param["wbp_train_iter"][idx]):
                with tf.GradientTape() as tape:
                    # Forward pass (generates fresh fading/noise every time)
                    b, llr_in, loss = self(batch_size, train_param["wbp_ebno_train"][idx])

                grads = tape.gradient(loss, self.trainable_variables)
                # Optional gradient clipping
                if "wbp_clip_value_grad" in train_param:
                     grads = [tf.clip_by_value(g, -train_param["wbp_clip_value_grad"], train_param["wbp_clip_value_grad"]) for g in grads]

                optimizer.apply_gradients(zip(grads, self.trainable_weights))

                if it % 100 == 0:
                    # Validation step
                    b_val, llr_val, val_loss = self(train_param["wbp_batch_size_val"], train_param["wbp_ebno_val"])
                    ber = compute_ber(b_val, hard_decisions(llr_val))
                    print(f"Iter {it}: Loss={loss.numpy():.4f}, Val BER={ber.numpy():.6f}")