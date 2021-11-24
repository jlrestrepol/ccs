#%%
import numpy as np
import pandas as pd
import sklearn.preprocessing as sk_pp
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import utils
#%%
def fit_encoder(data):   
    """Fits an encoder to a series of sequences"""
    data = [list(d) for d in data]#Create a nested list: list(str) -> [[char] for char in str]
    flat_list = ['_'] + [item for sublist in data for item in sublist]#flattens nested list
    values = np.array(flat_list)#initialize np array
    label_encoder = sk_pp.LabelEncoder()#Try OrdinalEncoder()
    label_encoder.fit(values)#fit encoder
    return label_encoder

def encoder(df, encoder, inplace = False):
    """Prepares the data and econdes it"""
    if not inplace:
        df_cp = df.copy()
    df_cp['Modified_sequence'] = df_cp['Modified_sequence'].str.replace('_','')
    df_cp['encseq'] = df_cp['Modified_sequence'].apply(lambda x: encoder.transform(list(x))) 
    return df_cp

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

model_params = {'timesteps': 66}

def counts_matrix(df):
    """Generates a matrix of counts starting from a csv file with sequence and charge"""
    data = df['Modified_sequence']
    label_encoder = fit_encoder(data)
    pp_data = encoder(df, label_encoder)
    encoded = int_dataset(pp_data, model_params['timesteps'], middle = False)
    encoded = encoded[:,:,0]

    counts_matrix = np.zeros((encoded.shape[0], 27), dtype=np.int8)
    for i, row in enumerate(encoded):
        index, counts = np.unique(row[:-1], return_counts=True)
        counts_matrix[i, index] = counts
    counts_matrix = np.append(counts_matrix, encoded[:,-1].reshape(-1,1), axis = 1)
    return counts_matrix

def predict(counts, df):
    """Loads in a counts matrix and predicts using XGBoost"""
    #m/z
    print('Calculating m/z of given sequences')
    df['m/z'] = df.apply(lambda x: utils.calculate_mass(x['Modified_sequence'], x['Charge']), axis = 1)
    #power law
    print('Calculating m/z-based ccs prediction')
    popt_2 = np.array([15.07814883,  0.49945974])
    popt_3 = np.array([35.65345317,  0.41790597])
    popt_4 = np.array([61.12222885,  0.38207705])

    f = lambda x, A, b: A*x**b 

    df_ch2 = df[df['Charge']==2]
    df_ch3 = df[df['Charge']==3]
    df_ch4 = df[df['Charge']==4]

    df.loc[df['Charge']==2,'predicted_ccs'] = f(df_ch2['m/z'], popt_2[0], popt_2[1])
    df.loc[df['Charge']==3,'predicted_ccs'] = f(df_ch3['m/z'], popt_3[0], popt_3[1])
    df.loc[df['Charge']==4,'predicted_ccs'] = f(df_ch4['m/z'], popt_4[0], popt_4[1])

    xgb_ch2 = joblib.load('../models/xgboost_count/xgb_counts_ch2')
    xgb_ch3 = joblib.load('../models/xgboost_count/xgb_counts_ch3')
    xgb_ch4 = joblib.load('../models/xgboost_count/xgb_counts_ch4')

    print('Calculating XGboost-based ccs prediction')
    df['xgboost'] = 0



    df.loc[df['Charge']==2,'xgboost'] = xgb_ch2.predict(counts[counts[:,-1]==2]) + df.loc[df['Charge']==2,'predicted_ccs']
    df.loc[df['Charge']==3,'xgboost'] = xgb_ch3.predict(counts[counts[:,-1]==3]) + df.loc[df['Charge']==3,'predicted_ccs'] 
    df.loc[df['Charge']==4,'xgboost'] = xgb_ch4.predict(counts[counts[:,-1]==4]) + df.loc[df['Charge']==4,'predicted_ccs']
# %%
df = pd.read_csv('../dl_paper/SourceData_Figure_4.csv').loc[:,['Modified_sequence', 'Charge', 'CCS']]
# %%
counts = counts_matrix(df)
#%%
predict(counts, df)
#%%
#df.to_csv('predictions.csv')
#%%
df_fig4 = pd.read_pickle('../Data/Fig4_powerlaw.pkl')
# %%
