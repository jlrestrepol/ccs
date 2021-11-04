#%%
import logging
import time
import pandas as pd
from sklearn import model_selection
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow_text as text
import tensorflow as tf
#%%
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
#%%
def train_val_set(charge = np.nan):
    """Function that outputs train and validation set for a given charge state"""
    fig1 = pd.read_pickle('/mnt/pool-cox-data08/Juan/ccs/Data/Fig1_powerlaw.pkl')#loads in raw training data
    features_complete = np.load('/mnt/pool-cox-data08/Juan/ccs/Data/one_hot_encoded_fig1.npy', allow_pickle=True)#load in one-hot-encoded training data
    label_complete = (fig1['CCS'] - fig1['predicted_ccs']).values#residual
    features = features_complete[features_complete[:,-1] == charge][:,:-1]#choose points with given charge, drop charge feature because of 2 heads
    label = label_complete[features_complete[:,-1] == charge]#choose appropiate residuals
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, label, test_size = 0.1, random_state=42)#train/val set split
    index = np.random.choice(x_train.shape[0], size = 150000, replace = False) #subsample
    x_train = x_train[index,:]
    y_train = y_train[index]
    print(f"The Initial Mean Squared Error is: {sk.metrics.mean_squared_error(fig1['CCS'], fig1['predicted_ccs'])}")#prints initial error
    del fig1
    del features_complete
    del label_complete
    del features
    del label
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
#%%
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)
# %%
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 22), tf.bool)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
# %%
def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)
# %%
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model = 512, num_heads = 8, causal=False, dropout=0.0):
    super(MultiHeadAttention, self).__init__()

    assert int(d_model) % int(num_heads) == 0
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

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, mask_zero=False)
    self.pos = positional_encoding(maximum_position_encoding, d_model)

    self.encoder_layers = [ EncoderLayer(d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout) for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(dropout)

  def call(self, inputs, mask=None, training=None):
    x = self.embedding(inputs)
    # positional encoding
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # scaling by the sqrt of d_model, not sure why or if needed??
    x += self.pos[: , :tf.shape(x)[1], :]

    x = self.dropout(x, training=training)

    #Encoder layer
    embedding_mask = create_padding_mask(inputs)
    for encoder_layer in self.encoder_layers:
      x = encoder_layer(x, mask = embedding_mask)

    return x

  def compute_mask(self, inputs, mask=None):
    return self.embedding.compute_mask(inputs)

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
  
  def get_config(self):
    config = {
    'd_model': self.d_model,
    'warmup_steps': self.warmup_steps,

     }
    return config
#%%
def test_set_results():
    '''Results on the complete test set'''
    prefix =  '/mnt/pool-cox-data08/Juan/ccs/models/transformer'
    model_ch2, _ = architecture()
    model_ch2.load_weights(prefix+'_ch2/checkpoints/best')
    model_ch3, _ = architecture()
    model_ch3.load_weights(prefix+'_ch3/checkpoints/best')
    model_ch4, _ = architecture()
    model_ch4.load_weights(prefix+'_ch4/checkpoints/best')

    prefix = '/mnt/pool-cox-data08/Juan/ccs/Data/'
    df_fig4 = pd.read_pickle(prefix+'Fig4_powerlaw.pkl')
    features_fig4 = np.load(prefix+'one_hot_encoded_fig4.npy', allow_pickle=True)

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
#%%
def architecture():
  num_layers = 4
  d_model = 128
  dff = 512
  num_heads = 8
  dropout_rate = 0.1
  input_vocab_size = 27+1
  target_vocab_size = 1
  #%%
  input = tf.keras.layers.Input(shape=(None,))
  target = tf.keras.layers.Input(shape=(None,))
  encoder = Encoder(input_vocab_size, num_layers = num_layers, d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout_rate)
  #decoder = Decoder(target_vocab_size, num_layers = num_layers, d_model = d_model, num_heads = num_heads, dff = dff, dropout = dropout_rate)

  x = encoder(input)
  x = tf.keras.layers.GlobalAveragePooling1D()(x)
  x = tf.keras.layers.Dropout(0.1)(x)
  x = tf.keras.layers.Dense(20, activation="relu")(x)
  x = tf.keras.layers.Dropout(0.1)(x)
  x = tf.keras.layers.Dense(1)(x)
  print(x.shape)

  model = tf.keras.models.Model(inputs=input, outputs=x)
  model.summary()
  optimizer = tf.keras.optimizers.Adam(CustomSchedule(d_model), beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
  return model, optimizer
#%%
loss = tf.keras.losses.MeanSquaredError(reduction = 'none')

def masked_loss(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 22))
  loss_ = loss(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

metrics = [loss, masked_loss]
#%%

model, optimizer = architecture()
model.compile(optimizer=optimizer, loss = loss, metrics = metrics) # masked_
#%%
charge = 2
x_train, x_test, y_train, y_test = train_val_set(charge = charge)
#%%
#folder = f'Y:\\Juan\\ccs\\models\\transformer_ch{charge}'
folder = f'..\\models\\transformer_ch{charge}'
cb = [tf.keras.callbacks.CSVLogger(folder+'\\training.log', append=False),  
        tf.keras.callbacks.ModelCheckpoint(folder+'\\checkpoints\\best', save_best_only=True, 
        save_weights_only = True)]

history = model.fit(
    x_train, y_train, batch_size=8, epochs=30, validation_data=(x_test, y_test), callbacks=cb
)
# %%
