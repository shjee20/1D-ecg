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

def basic_block(x, filters, kernel_size=3, strides=1):
    """ResNet 기본 블록 (3x3 Conv → 3x3 Conv)"""
    shortcut = x  # Identity mapping

    # 첫 번째 3x3 Conv
    x = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='same',
                               use_bias= True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # 두 번째 3x3 Conv
    x = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same',
                               use_bias= True)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # 스킵 커넥션: 입력과 출력의 채널 수 또는 스트라이드가 다를 경우 맞춰주기
    if shortcut.shape[-1] != filters or strides != 1:
        shortcut = tf.keras.layers.Conv1D(filters, kernel_size=1, strides=strides, padding='same',
                                         use_bias= True)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # Residual 연결
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)

    return x




def bottleneck_block(x, filters, kernel_size=3, strides=1):
    """ ResNet-50 Bottleneck Block (1x1 → 3x3 → 1x1 Conv) """
    shortcut = x  # Identity mapping

    # 1x1 Conv (축소 단계)
    x = tf.keras.layers.Conv1D(filters // 4, kernel_size=1, strides=1, padding='same',use_bias= True
                            )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # 3x3 Conv (Feature Extraction)
    x = tf.keras.layers.Conv1D(filters // 4, kernel_size=kernel_size, strides=strides, padding='same', use_bias= True
                            )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # 1x1 Conv (확장 단계)
    x = tf.keras.layers.Conv1D(filters, kernel_size=1, strides=1, padding='same', use_bias= True
                            )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Skip Connection (입력과 출력 차원 맞추기)
    if shortcut.shape[-1] != filters or strides != 1:
        shortcut = tf.keras.layers.Conv1D(filters, kernel_size=1, strides=strides, padding='same',use_bias= True
                                        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # Add residual connection
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)

    return x


        
'''ResNet50 '''

def encoder_resnet_1d(n_input, n_output, lr ):
    input_layer = tf.keras.layers.Input(shape=(n_input))

    # Initial Conv Layer
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=7, strides=2, padding='same', use_bias= True)(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)


# Stage 1 (3 Bottleneck Blocks, 256 filters)
    for i in range(3):
        x = bottleneck_block(x, filters=64, kernel_size=5, strides=2 if i == 0 else 1)

    # Stage 2 (4 Bottleneck Blocks, 512 filters, 첫 블록 strides=2)
    for i in range(4):
        x =bottleneck_block(x, filters=128, kernel_size=5, strides=2 if i == 0 else 1)
    
    # Stage 3 (6 Bottleneck Blocks, 1024 filters, 첫 블록 strides=2)
    for i in range(6):
        x =bottleneck_block(x, filters=256, kernel_size=5, strides=2 if i == 0 else 1)

    # Stage 4 (3 Bottleneck Blocks, 2048 filters, 첫 블록 strides=2)
    for i in range(3):
        x = bottleneck_block(x, filters=512, kernel_size=5, strides=2 if i == 0 else 1)

    # Global Average Pooling (GAP)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Fully Connected Layer
    x = tf.keras.layers.Dropout(rate=0.3)(x)   
    x = tf.keras.layers.Dense(units=200, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)  

    # Output Layer
    output_layer = tf.keras.layers.Dense(units=n_output, activation='sigmoid')(x)

    initial_thresholds = [0.13] * n_output
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    f1_metric = CustomF1WithClassThresholds(num_classes=n_output, thresholds=initial_thresholds, average='macro')
    optimizer = 'adam'

    # Compile Model
    '''thresshold 튜닝'''
    model.compile(
            optimizer=optimizer,
            loss=AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05), 
            metrics= [f1_metric] 
    
        )

    return model, f1_metric
