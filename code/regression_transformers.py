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

#%%
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
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

    def call(self, inputs):
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
# %%
############### Load in data and calculate error ####################
fig1 = pd.read_pickle('../Data/Fig1_powerlaw.pkl')
features_complete = np.load('../Data/one_hot_encoded_fig1.npy')
label_complete = (fig1['CCS'] - fig1['predicted_ccs']).values
charge = 2
features = features_complete[features_complete[:,-1] == charge][:,:-1]#drop charge
label = label_complete[features_complete[:,-1] == charge]
x_train, x_test, y_train, y_test = model_selection.train_test_split(features, label, test_size = 0.1, random_state=42)
print(f"The Initial Mean Absolute Error is: {sk.metrics.mean_absolute_error(fig1['CCS'], fig1['predicted_ccs'])}")

# %%
########### Predictor architecture ###########
embed_dim = x_train.shape[1]  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(embed_dim,))
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(inputs)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
# %%
############## Format input and fit model #############
X_train_tensor = np.asarray(x_train)
y_train_tensor = np.asarray(y_train)

X_test_tensor = np.asarray(x_test)
y_test_tensor = np.asarray(y_test)

#%%
print(len(X_train_tensor), "Training data")
print(len(X_test_tensor), "Test data")

model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(
    X_train_tensor, y_train_tensor, batch_size=64, epochs=15, validation_data=(X_test_tensor, y_test_tensor)
)
#%%
######### Diagnostic plots ###########
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
# %%
############## Load-in or save model ############

#!mkdir -p ../models/transformer_ch2 #create folder if doesnt exist
model.save('../models/transformer_ch2')



#model = tf.keras.models.load_model('transformer')
# %%
############ Test Set Results ############
df_fig4 = pd.read_pickle('../Data/Fig4_powerlaw.pkl')
features_fig4 = np.load('../Data/one_hot_encoded_fig4.npy')

##### Separated by charge
charge = 2
res  = model.predict(features_fig4[features_fig4[:,-1] == charge][:,:-1])
pred_ccs = df_fig4[df_fig4['Charge']==charge]['predicted_ccs'] + res.flatten()
# %%
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 6))
ax[0].hist((df_fig4[df_fig4['Charge']==charge]['CCS']-pred_ccs)/pred_ccs*100, bins = 50)
ax[0].set_xlabel('Relative Error of CCS')
ax[0].set_ylabel('Counts')
ax[0].set_title('Relative error of CCS w.r.t Ground Truth, Charge 2')

corr, _ = scipy.stats.pearsonr(df_fig4[df_fig4['Charge']==charge]['CCS'],pred_ccs)
ax[1].scatter(df_fig4[df_fig4['Charge']==charge]['CCS'],pred_ccs, label = f'Corr : {np.round(corr, 3)}', s = 0.1)
ax[1].set_xlabel('CCS')
ax[1].set_ylabel('Predicted CCS')
ax[1].set_title('Scatter Plot CCS vs predicted CCS, Charge 2')
ax[1].plot(np.arange(300,600), np.arange(300,600), 'b--')
ax[1].legend()
# %%
