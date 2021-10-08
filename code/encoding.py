#%%
import numpy as np
import pandas as pd
import sklearn.preprocessing as sk_pp

# %%
def prepare_data(df, encoder, inplace = False):
    """Prepares the data and econdes it"""
    if not inplace:
        df_cp = df.copy()
    df_cp = df.rename(index=str, columns={"Modified sequence": "Modified_sequence"})
    df_cp['Modified_sequence'] = df_cp['Modified_sequence'].str.replace('_','')
    df_cp['encseq'] = df_cp['Modified_sequence'].apply(lambda x: encoder.transform(list(x)))
    df_cp['len']=df_cp['Modified_sequence'].str.len()
    df_cp['minval'] = 275.41885 # np.min(df['CCS])
    df_cp['maxval'] = 1113.0214 # np.max(df['CCS])
    df_cp['CCS']=0
    df_cp['label']=df_cp['CCS'].values.tolist()
    
    return df_cp

def fit_encoder(data):   
    """Fits an encoder to a series of sequences"""
    data = [list(d) for d in data]#Create a nested list: list(str) -> [[char] for char in str]
    flat_list = ['_'] + [item for sublist in data for item in sublist]#flattens nested list
    values = np.array(flat_list)#initialize np array
    label_encoder = sk_pp.LabelEncoder()#Try OrdinalEncoder()
    label_encoder.fit(values)#fit encoder
    return label_encoder


def int_dataset(dat, timesteps, middle = True):
    """Returns matrix len(data) x timesteps, each entry is the aminoacid that goes in that position. Fixed Lenght = timesteps = 66.
       Shorter sequences are paded with '_' (encoded as 22)
    
    """
    empty_entry = 22
    oh_dat = (np.ones([len(dat), timesteps + 1, 1], dtype=np.int32)*empty_entry).astype(np.int32)
    cnt = 0
    for _, row in dat.iterrows():
        ie = np.array(row['encseq'])
        oe = ie.reshape(len(ie), 1)
        if middle:
            oh_dat[cnt, ((60-oe.shape[0])//2): ((60-oe.shape[0])//2)+oe.shape[0], :] = oe
        else:
            oh_dat[cnt, 0:oe.shape[0], :] = oe
        oh_dat[cnt, -1, 0] = row['Charge']
        cnt += 1

    return oh_dat

model_params = {"lab_name": "label", "fname": "cache/one_dat_cache_full_label.npy", "num_input": 32, "timesteps": 66,
 "num_hidden": 128, "num_layers": 2, "num_classes": 1, "dropout_keep_prob": 0.9, "use_uncertainty": False, 
 "use_attention": True, "simple": False, "num_tasks": -1, "batch_size": 64, 
 "model_dir": "out/Tests_200206_ward_min2_PTtest_2/", "model_dir_pretrain": "", "lr_base": 0.001, 
 "training_steps": 55000, "reduce_lr_step": 50000, "train_file": "data_final/Tests/200206_ward_min2_PTtest/2_train.pkl",
  "test_file": "data_final/Tests/200206_ward_min2_PTtest/2_test.pkl", "reduce_train": 0.5, 
  "scaling_dict": {"0": [275.41885375976557, 1118.7861328125]}}



#%%
########### Encode Sequence #############
df = pd.read_csv('../dl_paper/SourceData_Figure_1.csv').loc[:,['Modified sequence', 'Charge']]
data = df['Modified sequence']
label_encoder = fit_encoder(data)
#%%
pp_data = prepare_data(df, label_encoder)
encoded = int_dataset(pp_data, model_params['timesteps'], middle = False)
encoded = encoded[:,:,0]
encoded
#%%
np.save('../Data/encoded_fig1', encoded, allow_pickle=True)




# %%
############## Check That the encoding worked correctly ###########
#Fig1
#encoded = np.load('../Data/encoded_fig1.npy')[:,:-1]
decoded = ["".join(label_encoder.inverse_transform(e)) for e in encoded]
decoded = [e[:e.find('_')] for e in decoded]
f"Was the reconstruction perfect? {np.array_equal(df['Modified sequence'].str.replace('_','').to_list(), decoded)}"
# %%
#Fig4
df4 = pd.read_csv('../dl_paper/SourceData_Figure_4.csv')
encoded = np.load('../Data/encoded_fig4.npy')
decoded = ["".join(label_encoder.inverse_transform(e)) for e in encoded]
decoded = [e[:e.find('_')] for e in decoded]
f"Was the reconstruction perfect? {np.array_equal(df4['Modified_sequence'].str.replace('_','').to_list(), decoded)}"





############### Transform to one-hot-encoded ###################
# %%
encoded_fig1 = np.load('../Data/encoded_fig1.npy')
encoded_fig4 = np.load('../Data/encoded_fig4.npy')
oh_encoder = sk_pp.OneHotEncoder(sparse = False, categories = [np.arange(27)]*66, dtype=np.int8)
oh_encoder.fit(encoded_fig4[:,:-1])
ohe = oh_encoder.transform(encoded_fig4[:,:-1])
ohe = np.append(ohe, encoded_fig4[:,-1].reshape(-1,1), axis = 1)
#%%
np.save('../Data/one_hot_encoded_fig4', ohe, allow_pickle=True)

# %%
