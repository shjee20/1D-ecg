import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat
import pandas as pd
from scipy import signal
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, ProgbarLogger
from sklearn.model_selection import train_test_split
import ast 
import pickle
import matplotlib.pyplot as plt
import IPython.display as display
from tensorflow.keras.callbacks import TensorBoard
import datetime
from tqdm import tqdm
import random
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, LambdaCallback
import tensorflow.keras.backend as K
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from helper_code import AsymmetricLoss
from helper_code import CustomF1WithClassThresholds

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def call(self, q, k, v, mask=None):
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
        if mask is not None:
            scores += (mask * -1e9)
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, v), weights


class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(self, h, d_model, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.d_model = d_model
        self.q_linear = layers.Dense(d_model)
        self.k_linear = layers.Dense(d_model)
        self.v_linear = layers.Dense(d_model)
        self.out_proj = layers.Dense(d_model)
        self.attn = ScaledDotProductAttention()
        self.dropout = layers.Dropout(dropout)

    def get_config(self):
        return {
            "h": self.h,
            "d_model": self.d_model,
            "dropout": self.dropout.rate
        }

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.h, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask=None, training=False):
        B = tf.shape(q)[0]
        q = self.split_heads(self.q_linear(q), B)
        k = self.split_heads(self.k_linear(k), B)
        v = self.split_heads(self.v_linear(v), B)
        attn_out,  weights  = self.attn(q, k, v, mask)
        attn_out = tf.transpose(attn_out, [0, 2, 1, 3])
        concat = tf.reshape(attn_out, (B, -1, self.h * self.d_k))
        return self.dropout(self.out_proj(concat), training=training), weights


# class PositionwiseFeedForward(tf.keras.layers.Layer):
#     def __init__(self, d_model, d_ff, dropout=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.d_model = d_model
#         self.d_ff = d_ff
#         self.fc1 = layers.Dense(d_ff)
#         self.fc2 = layers.Dense(d_model)
#         self.dropout = layers.Dropout(dropout)

#     def get_config(self):
#         return {
#             "d_model": self.d_model,
#             "d_ff": self.d_ff,
#             "dropout": self.dropout.rate
#         }

#     def call(self, x, training=False):
#         x = gelu(self.fc1(x))
#         x = self.dropout(x, training=training)
#         return self.fc2(x)

# class SublayerConnection(tf.keras.layers.Layer):
#     def __init__(self, dropout=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.norm = layers.LayerNormalization(epsilon=1e-6)
#         self.dropout = layers.Dropout(dropout)

#     def get_config(self):
#         return {
#             "dropout": self.dropout.rate
#         }

#     def call(self, x, sublayer, training=False):
#         return x + self.dropout(sublayer(self.norm(x), training=training), training=training)


# class TransformerBlock(tf.keras.layers.Layer):
#     def __init__(self, hidden, attn_heads, ff_hidden, dropout=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.hidden = hidden
#         self.attn_heads = attn_heads
#         self.ff_hidden = ff_hidden
#         self.attn = MultiHeadedAttention(attn_heads, hidden, dropout)
#         self.ff = PositionwiseFeedForward(hidden, ff_hidden, dropout)
#         self.sublayer1 = SublayerConnection(dropout)
#         self.sublayer2 = SublayerConnection(dropout)
#         self.dropout = layers.Dropout(dropout)

#     def get_config(self):
#         return {
#             "hidden": self.hidden,
#             "attn_heads": self.attn_heads,
#             "ff_hidden": self.ff_hidden,
#             "dropout": self.dropout.rate
#         }

# def call(self, x, training=False):
#     x = self.sublayer1(x, lambda x_, training: self.attn(x_, x_, x_, training=training), training=training)
#     x = self.sublayer2(x, lambda x_, training: self.ff(x_, training=training), training=training)
#     return self.dropout(x, training=training)

# class TRANSFORMER(tf.keras.Model):
#     def __init__(self, hidden=128, n_layers=6, attn_heads=8, dropout=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.hidden = hidden
#         self.n_layers = n_layers
#         self.attn_heads = attn_heads
#         self.dropout = dropout
#         self.pe = PositionalEncoding(hidden)
#         self.blocks = [TransformerBlock(hidden, attn_heads, 4 * hidden, dropout) for _ in range(n_layers)]

#     def get_config(self):
#         return {
#             "hidden": self.hidden,
#             "n_layers": self.n_layers,
#             "attn_heads": self.attn_heads,
#             "dropout": self.dropout
#         }

#     def call(self, x, training=False):
#         x = self.pe(x)
#         for block in self.blocks:
#             x = block(x, training=training)
#         return x


# # --- Full Model Definition ---
# def cnn_transformer_encoder_model(n_input=(5000, 12), n_output=23, lr=0.001):
#     cnn_channels = 256
#     hidden_size = 256
#     ff_dim = 1024
#     num_heads = 8
#     num_encoder_layers = 8 
#     dropout = 0.1

#     inputs = tf.keras.Input(shape=n_input)  # (batch, time, channels)

#     # CNN Encoder
#     x = conv_block(inputs, 32, 7, 2, dropout=dropout)
#     x = conv_block(x, 64, 5, 2, dropout=dropout)
#     x = conv_block(x, 128, 5, 2, dropout=dropout)
#     x = conv_block(x, 128, 3, 2, dropout=dropout)
#     x = conv_block(x, 128, 3, 2, dropout=dropout)
#     x = conv_block(x, cnn_channels, 3, 2, dropout=dropout) 

#     # Transformer Encoder (positional encoding 포함)
#     transformer_encoder = TRANSFORMER(
#         hidden=hidden_size,
#         n_layers=num_encoder_layers,
#         attn_heads=num_heads,
#         dropout=dropout
#     )
#     x = transformer_encoder(x)  # Output: (batch, seq_len, hidden_size)

#     # Classification Head: Global pooling + Dense output
#     x = tf.keras.layers.GlobalAveragePooling1D()(x)       # (batch, hidden)
#     x = tf.keras.layers.Dense(n_output)(x)                # (batch, n_output)
#     outputs = tf.keras.layers.Activation("sigmoid")(x)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ECG_Transformer_Encoder")

#     # Optimizer
#     optimizer = AdamW(learning_rate=lr, weight_decay=0.0001)

#     # Compile model
#     model.compile(
#         optimizer=optimizer,
#         loss=AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05),
#         metrics=[tfa.metrics.F1Score(num_classes=n_output, average='macro', threshold=0.4)]
#     )

#     return model

CONFIG = {
    "d_model": 192,
    "ff_dim": 768,
    "num_heads": 6,
    "num_layers": 6,
    "dropout": 0.1,
    "mask_ratio": 0.25
}

# --- Positional Embedding (Learnable) ---
class LearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=[1, max_len, d_model],
            initializer="random_normal",
            trainable=True
        )

    def call(self, x):
        return x + self.pos_embedding[:, :tf.shape(x)[1], :]

    def get_config(self):
        return {"max_len": self.max_len, "d_model": self.d_model}

# --- Patch Masking ---
class PatchMasking(tf.keras.layers.Layer):
    def __init__(self, mask_ratio=CONFIG["mask_ratio"]):
        super().__init__()
        self.mask_ratio = mask_ratio

    def call(self, x):
        B, T, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        num_mask = tf.cast(tf.cast(T, tf.float32) * self.mask_ratio, tf.int32)
        rand_indices = tf.argsort(tf.random.uniform((B, T)), axis=-1)

        batch_indices = tf.reshape(tf.tile(tf.expand_dims(tf.range(B), 1), [1, num_mask]), [-1])
        mask_indices = tf.reshape(rand_indices[:, :num_mask], [-1])
        keep_indices = tf.reshape(rand_indices[:, num_mask:], [-1])

        flat = tf.reshape(x, [B * T, D])
        masked = tf.gather(flat, mask_indices + batch_indices * T)
        unmasked = tf.gather(flat, keep_indices + batch_indices * T)

        return tf.reshape(unmasked, [B, -1, D]), tf.reshape(masked, [B, -1, D]), rand_indices

    def get_config(self):
        return {"mask_ratio": self.mask_ratio}

# --- Conv Block for Feature Extraction ---
def conv_block(x, filters, kernel, stride, dropout=0.0):
    x = tf.keras.layers.Conv1D(filters, kernel_size=kernel, strides=stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    return tf.keras.layers.ReLU()(x)

# --- Conv Embedding ---
def conv_embedding(x, d_model):
    x = conv_block(x, filters=64, kernel=7, stride=2, dropout=0.1)
    x = conv_block(x, filters=128, kernel=5, stride=2, dropout=0.1)
    x = conv_block(x, filters=192, kernel=3, stride=2, dropout=0.1)
    x = conv_block(x, filters=192, kernel=3, stride=2, dropout=0.1)
    return tf.keras.layers.Dense(d_model)(x)

# --- Transformer Block ---
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.att = MultiHeadedAttention(h=num_heads, d_model=d_model, dropout=dropout)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='gelu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False, mask=None, return_attention=False):

        attn_output, attn_weights = self.att(x, x, x, mask=mask, training=training)
        x = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(x)
        output = self.layernorm2(x + self.dropout2(ffn_output, training=training))

        if return_attention:
            return output, attn_weights
        else:
            return output

    def get_config(self):
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate
        }
# --- MAE Encoder ---
class MAEEncoder(tf.keras.Model):
    def __init__(self, num_layers=CONFIG["num_layers"], d_model=CONFIG["d_model"], num_heads=CONFIG["num_heads"], ff_dim=CONFIG["ff_dim"], dropout=CONFIG["dropout"]):
        super().__init__()
        self.pos_embed = None  # Will be created dynamically based on input
        self.blocks = [TransformerBlock(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)]

    def call(self, x, training=False):
        x = conv_embedding(x, CONFIG["d_model"])
        if self.pos_embed is None:
            self.pos_embed = LearnablePositionalEncoding(max_len=tf.shape(x)[1], d_model=CONFIG["d_model"])
        x = self.pos_embed(x)
        for blk in self.blocks:
            x = blk(x, training=training)
        return x

    def get_config(self):
        return CONFIG

# --- MAE Decoder ---
class MAEDecoder(tf.keras.Model):
    def __init__(self, d_model=CONFIG["d_model"], num_heads=CONFIG["num_heads"], ff_dim=CONFIG["ff_dim"], num_layers=1, dropout=CONFIG["dropout"]):
        super().__init__()
        self.blocks = [TransformerBlock(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        self.output_proj = tf.keras.layers.Dense(d_model)

    def call(self, x, training=False):
        for blk in self.blocks:
            x = blk(x, training=training)
        return self.output_proj(x)

    def get_config(self):
        return {
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate
        }

# --- MAE Pretraining Model ---
class MAEPretrainModel(tf.keras.Model):
    def __init__(self, encoder, decoder, mask_ratio=CONFIG["mask_ratio"]):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_layer = PatchMasking(mask_ratio)

    def call(self, x, training=False):
        unmasked, masked, _ = self.mask_layer(x)
        encoded = self.encoder(unmasked, training=training)
        padded_input = tf.concat([encoded, tf.zeros_like(masked)], axis=1)
        decoded = self.decoder(padded_input, training=training)
        return decoded, masked

    def save_encoder_weights(self, path="mae_encoder_weights.h5"):
        self.encoder.save_weights(path)

    def load_encoder_weights(self, path="mae_encoder_weights.h5"):
        self.encoder.load_weights(path)

    def get_config(self):
        return CONFIG

# --- Fine-tuning Model ---
def mtecg_style_model(n_input=(5000, 12), n_output=23, lr=0.001, pretrained_encoder=None):
    inputs = tf.keras.Input(shape=n_input)
    x = conv_embedding(inputs, d_model=CONFIG["d_model"])
    max_len = n_input[0] // 16  + 1# infer max_len based on 4 conv layers with stride 2
    x = LearnablePositionalEncoding(max_len=max_len, d_model=CONFIG["d_model"])(x)

    if pretrained_encoder is not None:
        x = pretrained_encoder(x)
    else:
        for _ in range(CONFIG["num_layers"]):
            x = TransformerBlock(CONFIG["d_model"], CONFIG["num_heads"], CONFIG["ff_dim"], dropout=CONFIG["dropout"])(x)
            

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(n_output, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MTECG_Like_Transformer")
    initial_thresholds = [0.15] * n_output
    f1_metric = CustomF1WithClassThresholds(num_classes=n_output, thresholds=initial_thresholds, average='macro')
    optimizer = 'adam'
    model.compile(
        optimizer=optimizer,
        loss=AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05),
        metrics=[f1_metric]
    )

    return model

# class PositionalEncoding(layers.Layer):
#     def __init__(self, sequence_len, d_model, **kwargs):
#         super().__init__(**kwargs)
#         self.sequence_len = sequence_len
#         self.d_model = d_model

#         pos = tf.range(sequence_len, dtype=tf.float32)[:, tf.newaxis]
#         i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
#         angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
#         angle_rads = pos * angle_rates
#         sines = tf.math.sin(angle_rads[:, 0::2])
#         cosines = tf.math.cos(angle_rads[:, 1::2])
#         pos_encoding = tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]
#         self.pos_encoding = tf.cast(pos_encoding, tf.float32)

#     def call(self, x):
#         seq = tf.shape(x)[1]
#         return x + self.pos_encoding[:, :seq, :]

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "sequence_len": self.sequence_len,
#             "d_model": self.d_model,
#         })
#         return config

# class SublayerConnection(layers.Layer):
#     def __init__(self, hidden_size, dropout=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.hidden_size = hidden_size
#         self.dropout_rate = dropout
#         self.norm = layers.LayerNormalization(epsilon=1e-6)
#         self.dropout = layers.Dropout(dropout)

#     def call(self, x, sublayer):
#         out = self.norm(x)
#         out = sublayer(out)
#         out = self.dropout(out)
#         return x + out

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "hidden_size": self.hidden_size,
#             "dropout": self.dropout_rate,
#         })
#         return config

# class TransformerEncoderBlock(layers.Layer):
#     def __init__(self, hidden_size, ff_dim, num_heads=8, dropout=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.hidden_size = hidden_size
#         self.ff_dim = ff_dim
#         self.num_heads = num_heads
#         self.dropout_rate = dropout

#         self.attention = layers.MultiHeadAttention(
#             num_heads=num_heads,
#             key_dim=hidden_size // num_heads,
#             dropout=dropout
#         )
#         self.attn_sublayer = SublayerConnection(hidden_size, dropout)
#         self.ff_sublayer = SublayerConnection(hidden_size, dropout)
#         self.ffn = tf.keras.Sequential([
#             layers.Dense(ff_dim, activation='relu'),
#             layers.Dense(hidden_size),
#             layers.Dropout(dropout)
#         ])

#     def call(self, x):
#         x = self.attn_sublayer(x, lambda x_: self.attention(x_, x_))
#         x = self.ff_sublayer(x, self.ffn)
#         return x

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "hidden_size": self.hidden_size,
#             "ff_dim": self.ff_dim,
#             "num_heads": self.num_heads,
#             "dropout": self.dropout_rate,
#         })
#         return config

# class TransformerEncoder(layers.Layer):
#     def __init__(self, num_layers, hidden_size, ff_dim, num_heads=8, dropout=0.1, max_len=5000, **kwargs):
#         super().__init__(**kwargs)
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.ff_dim = ff_dim
#         self.num_heads = num_heads
#         self.dropout_rate = dropout
#         self.max_len = max_len

#         self.pos_encoding = PositionalEncoding(sequence_len=max_len, d_model=hidden_size)
#         self.encoder_blocks = [
#             TransformerEncoderBlock(hidden_size, ff_dim, num_heads, dropout)
#             for _ in range(num_layers)
#         ]

#     def call(self, x):
#         x = self.pos_encoding(x)
#         for block in self.encoder_blocks:
#             x = block(x)
#         return x

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "num_layers": self.num_layers,
#             "hidden_size": self.hidden_size,
#             "ff_dim": self.ff_dim,
#             "num_heads": self.num_heads,
#             "dropout": self.dropout_rate,
#             "max_len": self.max_len,
#         })
#         return config
# --- Positional Encoding (Learnable) ---

# class PositionalEncoding(tf.keras.layers.Layer):
#     def __init__(self, d_model, dropout=0.1, max_len=5000, **kwargs):
#         super(PositionalEncoding, self).__init__(**kwargs)
#         self.d_model = d_model
#         self.max_len = max_len
#         self.dropout = tf.keras.layers.Dropout(rate=dropout)

#         # Compute the positional encoding matrix
#         pos = np.arange(max_len)[:, np.newaxis]               # (max_len, 1)
#         i = np.arange(d_model)[np.newaxis, :]                 # (1, d_model)
#         angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
#         angle_rads = pos * angle_rates                        # (max_len, d_model)

#         # Apply sin to even indices in the array; cos to odd indices
#         pos_encoding = np.zeros((max_len, d_model))
#         pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
#         pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

#         # Add batch dimension
#         pos_encoding = pos_encoding[np.newaxis, ...]          # (1, max_len, d_model)
#         self.pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)

#     def call(self, x, training=False):
#         seq_len = tf.shape(x)[1]
#         x = x + self.pos_encoding[:, :seq_len, :]
#         return self.dropout(x, training=training)

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "d_model": self.d_model,
#             "max_len": self.max_len,
#             "dropout": self.dropout.rate,
#         })
#         return config


# # --- Conv1D Block ---
# def conv_block(x, filters, kernel, stride, dropout=0.0):
#     x = tf.keras.layers.Conv1D(filters, kernel_size=kernel, strides=stride, padding="same", use_bias=False)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     if dropout > 0:
#         x = tf.keras.layers.Dropout(dropout)(x)
#     return tf.keras.layers.ReLU()(x)
