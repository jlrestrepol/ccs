#%%
import collections
import logging
import os
import pathlib
import re
import string
import sys
import time
import pandas as pd
from sklearn import model_selection
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import validation

#import tensorflow_text as text
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.gen_array_ops import Transpose
# %%
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
#%%
def train_val_set(charge = np.nan):
    """Function that outputs train and validation set for a given charge state"""
    fig1 = pd.read_pickle('../Data/Fig1_powerlaw.pkl')#loads in raw training data
    features_complete = np.load('/mnt/pool-cox-data08/Juan/ccs/Data/encoded_fig1.npy', allow_pickle=True)#load in one-hot-encoded training data
    label_complete = (fig1['CCS'] - fig1['predicted_ccs']).values#residual
    features = features_complete[features_complete[:,-1] == charge][:,:-1]#choose points with given charge, drop charge feature because of 2 heads
    label = label_complete[features_complete[:,-1] == charge]#choose appropiate residuals
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, label, test_size = 0.1, random_state=42)#train/val set split
    print(f"The Initial Mean Squared Error is: {sk.metrics.mean_squared_error(fig1['CCS'], fig1['predicted_ccs'])}")#prints initial error
    del fig1
    del features_complete
    del label_complete
    del features
    del label
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
# %%

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles( np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)
# %%
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 22), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
# %%
def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)
# %%
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,  d_model = 512, num_heads = 8, dff = 2048, dropout = 0.0):
    super(EncoderLayer, self).__init__()

    self.multi_head_attention =  MultiHeadAttention(d_model, num_heads)
    self.dropout_attention = tf.keras.layers.Dropout(dropout)
    self.add_attention = tf.keras.layers.Add()
    self.layer_norm_attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
    self.dense2 = tf.keras.layers.Dense(d_model)
    self.dropout_dense = tf.keras.layers.Dropout(dropout)
    self.add_dense = tf.keras.layers.Add()
    self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, inputs, mask=None, training=None):
    # print(mask)
    attention = self.multi_head_attention([inputs,inputs,inputs], mask = [mask,mask])
    attention = self.dropout_attention(attention, training = training)
    x = self.add_attention([inputs , attention])
    x = self.layer_norm_attention(x)
    # x = inputs

    ## Feed Forward
    dense = self.dense1(x)
    dense = self.dense2(dense)
    dense = self.dropout_dense(dense, training = training)
    x = self.add_dense([x , dense])
    x = self.layer_norm_dense(x)

    return x
# %%
class Encoder(tf.keras.layers.Layer):
  def __init__(self, input_vocab_size, num_layers = 4, d_model = 512, num_heads = 8, dff = 2048, maximum_position_encoding = 10000, dropout = 0.0):
    super(Encoder, self).__init__()

    self.d_model = d_model

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, mask_zero=True)
    self.pos = positional_encoding(maximum_position_encoding, d_model)

    self.encoder_layers = [ EncoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout) for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, inputs, mask=None, training=None):
    x = self.embedding(inputs)
    # positional encoding
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) 
    x += self.pos[: , :tf.shape(x)[1], :]

    x = self.dropout(x, training=training)

    #Encoder layer
    embedding_mask = self.embedding.compute_mask(inputs)
    for encoder_layer in self.encoder_layers:
      x = encoder_layer(x, mask = embedding_mask)

    return x

  def compute_mask(self, inputs, mask=None):
    return self.embedding.compute_mask(inputs)
# %%
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model = 512, num_heads = 8, causal=False, dropout=0.0):
    super(MultiHeadAttention, self).__init__()

    assert d_model % num_heads == 0
    depth = d_model // num_heads

    self.w_query = tf.keras.layers.Dense(d_model)
    self.split_reshape_query = tf.keras.layers.Reshape((-1,num_heads,depth))  
    self.split_permute_query = tf.keras.layers.Permute((2,1,3))      

    self.w_value = tf.keras.layers.Dense(d_model)
    self.split_reshape_value = tf.keras.layers.Reshape((-1,num_heads,depth))
    self.split_permute_value = tf.keras.layers.Permute((2,1,3))

    self.w_key = tf.keras.layers.Dense(d_model)
    self.split_reshape_key = tf.keras.layers.Reshape((-1,num_heads,depth))
    self.split_permute_key = tf.keras.layers.Permute((2,1,3))

    self.attention = tf.keras.layers.Attention(causal=causal, dropout=dropout)
    self.join_permute_attention = tf.keras.layers.Permute((2,1,3))
    self.join_reshape_attention = tf.keras.layers.Reshape((-1,d_model))

    self.dense = tf.keras.layers.Dense(d_model)

  def call(self, inputs, mask=None, training=None):
    q = inputs[0]
    v = inputs[1]
    k = inputs[2] if len(inputs) > 2 else v

    query = self.w_query(q)
    query = self.split_reshape_query(query)    
    query = self.split_permute_query(query)                 

    value = self.w_value(v)
    value = self.split_reshape_value(value)
    value = self.split_permute_value(value)

    key = self.w_key(k)
    key = self.split_reshape_key(key)
    key = self.split_permute_key(key)

    if mask is not None:
      if mask[0] is not None:
        mask[0] = tf.keras.layers.Reshape((-1,1))(mask[0])
        mask[0] = tf.keras.layers.Permute((2,1))(mask[0])
      if mask[1] is not None:
        mask[1] = tf.keras.layers.Reshape((-1,1))(mask[1])
        mask[1] = tf.keras.layers.Permute((2,1))(mask[1])

    attention = self.attention([query, value, key], mask=mask)
    attention = self.join_permute_attention(attention)
    attention = self.join_reshape_attention(attention)

    x = self.dense(attention)

    return x
# %%
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
# %%
# Hyperparameters
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

# Size of input vocab plus start and end tokens
input_vocab_size = 27
target_vocab_size = 1

input = tf.keras.layers.Input(shape=(None,))
target = tf.keras.layers.Input(shape=(None,))
encoder = Encoder(input_vocab_size, num_layers = num_layers, d_model = d_model, 
num_heads = num_heads, dff = dff, dropout = dropout_rate)

x = encoder(input)

x = tf.keras.layers.Dense(target_vocab_size)(x)
model = tf.keras.models.Model(inputs=[input, target], outputs=x)
model.summary()
#%%
x_train, x_test, y_train, y_test = train_val_set(charge = 2)
# %%
optimizer = tf.keras.optimizers.Adam(CustomSchedule(d_model), beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
loss_object = tf.keras.losses.MeanSquaredError(reduction='none')
model.compile(loss=loss_object, optimizer=optimizer)
# %%
model.fit([x_train, y_train], [y_train, x_train], batch_size=64, epochs=10,
              validation_data=([x_test, y_test], [y_test, x_test]))
        
# %%