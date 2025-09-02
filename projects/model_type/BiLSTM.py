import numpy as np, os, sys, joblib
from helper_code import DynamicF1
from helper_code import CustomF1WithClassThresholds
from helper_code import AsymmetricLoss
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

import tensorflow as tf

CONFIG = {
    "d_model": 256,
    "ff_dim": 1024,
    "num_heads": 8,
    "num_layers": 4,
    "dropout": 0.2,
    "conv_dropout": 0.1,
    "lstm_dropout": 0.2,
}

# Simple convolutional block (no residual)
def conv_block(x, filters, kernel, stride, dropout=0.0):
    x = tf.keras.layers.Conv1D(filters, kernel_size=kernel, strides=stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    return x

# Convolutional embedding without residual
def conv_embedding(x, d_model):
    x = conv_block(x, 64, 7, stride=2, dropout=CONFIG["conv_dropout"])
    x = conv_block(x, 128, 5, stride=2, dropout=CONFIG["conv_dropout"])
    x = conv_block(x, 256, 3, stride=1, dropout=CONFIG["conv_dropout"])
    x = conv_block(x, 256, 3, stride=1, dropout=CONFIG["conv_dropout"])
    return tf.keras.layers.Dense(d_model)(x)

# Global average pooling instead of multi-head attention
def simple_attention_pooling(x):
    return tf.keras.layers.GlobalAveragePooling1D()(x)

# Model with LSTM followed by BiLSTM
def build_sequential_lstm_model(input_shape, d_model):
    inputs = tf.keras.Input(shape=input_shape, name="input_signal")
    x = conv_embedding(inputs, d_model)
    x = tf.keras.layers.LSTM(128, return_sequences=True, dropout=CONFIG["lstm_dropout"])(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=CONFIG["lstm_dropout"]))(x)
    x = simple_attention_pooling(x)
    return inputs, x

# Final model assembly
def bilstm_style_model(n_input=(5000, 12), n_output=23, lr=0.001):
    inputs, x = build_sequential_lstm_model(n_input, CONFIG["d_model"])
    outputs = tf.keras.layers.Dense(n_output, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Sequential_LSTM_BiLSTM_Model")

    initial_thresholds = [0.15] * n_output
    f1_metric = CustomF1WithClassThresholds(num_classes=n_output, thresholds=initial_thresholds, average='macro')
    optimizer = 'adam'

    model.compile(
        optimizer=optimizer,
        loss=AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05),
        metrics=[f1_metric]
    )
    return model, f1_metric