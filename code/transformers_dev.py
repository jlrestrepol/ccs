#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn as sk
from sklearn import model_selection
import scipy
import sys

#%%
class MultiHeadAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, mask = None):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask = None):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
#%%
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
    attention = self.multi_head_attention(inputs, mask = None)
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

def diagnostic_plot(file_path):
    '''Diagnostic plots of the DL model'''
    data = pd.read_csv(file_path)
    plt.plot(data['loss'])
    plt.plot(data['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

def test_set_one_charge_results(charge = 2):
    '''Plots in the test set: scatter and histogram'''
    ### Load-in model and Data
    df_fig4 = pd.read_pickle('../Data/Fig4_powerlaw.pkl')
    features_fig4 = np.load('../Data/one_hot_encoded_fig4.npy')
    path = f'../models/transformer_ch{charge}/checkpoints/best'
    model = tf.keras.models.load_model(path)
    ### Separate by charge and make predictions, change after all the models are trained
    res  = model.predict(features_fig4[features_fig4[:,-1] == charge][:,:-1])
    pred_ccs = df_fig4[df_fig4['Charge']==charge]['predicted_ccs'] + res.flatten()
    err_rel = (pred_ccs - df_fig4[df_fig4['Charge']==charge]['CCS'])/pred_ccs*100
    ### Histogram and scatter plot
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 6))
    ax[0].hist(err_rel, bins = 50, label = f'MAD = {np.round(scipy.stats.median_abs_deviation(err_rel), 4)}')
    ax[0].set_xlabel('Relative Error of CCS')
    ax[0].set_ylabel('Counts')
    ax[0].set_title('Relative error of CCS w.r.t Ground Truth, Charge 2')
    ax[0].legend()

    corr, _ = scipy.stats.pearsonr(df_fig4[df_fig4['Charge']==charge]['CCS'],pred_ccs)
    ax[1].scatter(df_fig4[df_fig4['Charge']==charge]['CCS'],pred_ccs, label = f'Corr : {np.round(corr, 3)}', s = 0.1)
    ax[1].set_xlabel('CCS')
    ax[1].set_ylabel('Predicted CCS')
    ax[1].set_title('Scatter Plot CCS vs predicted CCS, Charge 2')
    ax[1].plot(np.arange(300,600), np.arange(300,600), 'b--')
    ax[1].legend()

def test_set_results():
    '''Results on the complete test set'''
    prefix = '../models/transformer'
    model_ch2 = tf.keras.models.load_model(prefix+'_ch2/checkpoints/best')
    model_ch3 = tf.keras.models.load_model(prefix+'_ch3/checkpoints/best')
    model_ch4 = tf.keras.models.load_model(prefix+'_ch4/checkpoints/best')

    df_fig4 = pd.read_pickle('../Data/Fig4_powerlaw.pkl')
    features_fig4 = np.load('../Data/one_hot_encoded_fig4.npy', allow_pickle=True)

    df_fig4['transformer'] = 0
    df_fig4.loc[df_fig4['Charge']==2,'transformer'] = model_ch2.predict(features_fig4[features_fig4[:,-1]==2][:,:-1]).flatten() + df_fig4.loc[df_fig4['Charge']==2,'predicted_ccs']
    df_fig4.loc[df_fig4['Charge']==3,'transformer'] = model_ch3.predict(features_fig4[features_fig4[:,-1]==3][:,:-1]).flatten() + df_fig4.loc[df_fig4['Charge']==3,'predicted_ccs']
    df_fig4.loc[df_fig4['Charge']==4,'transformer'] = model_ch4.predict(features_fig4[features_fig4[:,-1]==4][:,:-1]).flatten() + df_fig4.loc[df_fig4['Charge']==4,'predicted_ccs']

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 6))
    res_rel = (df_fig4['CCS']-df_fig4['transformer'])/df_fig4['transformer']*100
    ax[0].hist(res_rel, bins = 50, label = f'MAD = {np.round(scipy.stats.median_abs_deviation(res_rel), 4)}')
    ax[0].set_xlabel('Relative Error of CCS')
    ax[0].set_ylabel('Counts')
    ax[0].set_title('Relative error of CCS w.r.t Ground Truth')
    ax[0].legend()

    corr, _ = scipy.stats.pearsonr(df_fig4['transformer'],df_fig4['CCS'])
    print(f'The correlation is: {corr}')
    ax[1].scatter(df_fig4['CCS'], df_fig4['transformer'], label = f'Corr : {np.round(corr, 5)}', s = 0.1)
    ax[1].set_xlabel('CCS')
    ax[1].set_ylabel('Predicted CCS')
    ax[1].set_title('Scatter Plot CCS vs predicted CCS')
    ax[1].plot(np.arange(300,800), np.arange(300,800), 'b--')
    ax[1].legend()

def train_val_set(charge = np.nan):
    """Function that outputs train and validation set for a given charge state"""
    fig1 = pd.read_pickle('../Data/Fig1_powerlaw.pkl')#loads in raw training data
    features_complete = np.load('../Data/one_hot_encoded_fig1.npy')#load in one-hot-encoded training data
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
    return x_train, x_test, y_train, y_test

def architecutre(x_train):
    """Architecture of the predictor"""
    embed_dim = x_train.shape[1]  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    input_vocab_size = 27

    inputs = layers.Input(shape=(embed_dim,))
    '''transformer_block1 = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block1(inputs)
    print(x.shape, x_train.shape)
    transformer_block2 = TransformerBlock(x_train.shape[1], num_heads, ff_dim)
    x = transformer_block2(x)'''
    #transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    #x = transformer_block(inputs)
    encoder = Encoder(input_vocab_size, num_layers = num_layers, d_model = d_model, 
    num_heads = num_heads, dff = dff, dropout = dropout_rate)
    x = encoder(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

#%%
def train():
    charge = 3
    x_train, x_test, y_train, y_test = train_val_set(charge = charge)
    X_train_tensor = np.asarray(x_train)
    y_train_tensor = np.asarray(y_train)
    X_test_tensor = np.asarray(x_test)
    y_test_tensor = np.asarray(y_test)

    model = architecutre(x_train)#initializes new model
    #model = tf.keras.models.load_model('../models/transformer_ch2')#loads existing model
   
    ############## Format input and fit model #############

    folder = f'../models/transformer_ch{charge}/'
    cb = [tf.keras.callbacks.CSVLogger(folder+'training.log', append=False),  
         tf.keras.callbacks.ModelCheckpoint(folder+'checkpoints/best', save_best_only=True)]

    print(len(X_train_tensor), "Training data")
    print(len(X_test_tensor), "Test data")

    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(
        X_train_tensor, y_train_tensor, batch_size=64, epochs=30, validation_data=(X_test_tensor, y_test_tensor), callbacks=cb
    )

#%%
if __name__ == "__main__":
    train()
#%%
diagnostic_plot('../models/transformer_ch2/training.log')
# %%
