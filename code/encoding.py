#%%
import numpy as np
import pandas as pd
import sklearn.preprocessing as sk_pp
from sklearn.feature_extraction.text import CountVectorizer
import paths

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


########### Encode Sequence #############
def encode_seq(check = True, save = False):
    """Reads-in the raw data, fits a encoder to it, transforms it and saves it"""
    prefix_data = paths.get_paths()['data']
    df = pd.read_csv(prefix_data+'SourceData_Figure_1.csv').loc[:,['Modified sequence', 'Charge']]
    data = df['Modified sequence']
    label_encoder = fit_encoder(data)#I dont specify the vocabulary, but every aa/symbol should be present anyway
    pp_data = prepare_data(df, label_encoder)
    encoded = int_dataset(pp_data, model_params['timesteps'], middle = False)
    encoded = encoded[:,:,0]
    
    if check:#check that the encoding worked correctly
        decoded = ["".join(label_encoder.inverse_transform(e)) for e in encoded]
        decoded = [e[:e.find('_')] for e in decoded]
        print(f"Was the reconstruction perfect? {np.array_equal(df['Modified sequence'].str.replace('_','').to_list(), decoded)}")

    if save:#save the matrix
        np.save(prefix_data+'encoded_fig1', encoded, allow_pickle=True)

    return encoded

############### Transform to one-hot-encoded ###################
def ohe_seq(save = False):
    """Reads-in the encoded sequence, fits a ohe on the training data with specified categories and transform the data"""
    prefix_data = paths.get_paths()['data']
    encoded_fig1 = np.load(prefix_data+'encoded_fig1.npy')
    encoded_fig4 = np.load(prefix_data+'encoded_fig4.npy')
    oh_encoder = sk_pp.OneHotEncoder(sparse = False, categories = [np.arange(27)]*66, dtype=np.int8)#Initializes ohe with given classes
    oh_encoder.fit(encoded_fig1[:,:-1])#Fit the ohe in the training data without the charge
    ohe = oh_encoder.transform(encoded_fig4[:,:-1])#Transform training/test data
    ohe = np.append(ohe, encoded_fig4[:,-1].reshape(-1,1), axis = 1)#Adds the charge
    
    if save:#saves the matrix
        np.save(prefix_data+'/one_hot_encoded_fig4', ohe, allow_pickle=True)
    
    return ohe

############ Transfor to Count Matrix ##############
def count_seq(save = False):
    """Read-in the ohe sequences and creates from it a count matrix of aa"""
    prefix_data = paths.get_paths()['data']
    encoded_fig1 = np.load(prefix_data+'encoded_fig1.npy')
    counts_matrix = np.zeros((encoded_fig1.shape[0], 27), dtype=np.int8)
    for i, row in enumerate(encoded_fig1):
        index, counts = np.unique(row[:-1], return_counts=True)
        counts_matrix[i, index] = counts
    counts_matrix = np.append(counts_matrix, encoded_fig1[:,-1].reshape(-1,1), axis = 1)#appends charge

    if save:
        np.save(prefix_data+'counts_fig1', counts_matrix, allow_pickle=True)
    
    return counts_matrix

########## Dipeptides ############
def dipeptides_seq(save = False):
    """Read-in raw data, create vocabulary of dipeptides, fit a CountVectorizer object to get count of dipeptides"""
    
    prefix_data = paths.get_paths()['data']
    df = pd.read_csv(prefix_data+'SourceData_Figure_1.csv').loc[:,['Modified sequence', 'Charge']]
    seqs = df['Modified sequence'].str.replace('_','')
    aa = ['(', ')', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M','N', 'P', 'Q', 'R', 'S', 
    'T', 'V', 'W', 'Y', 'a', 'c', 'o','x']
    voc = [l2+l1 for l1 in aa for l2 in aa]#Creates vocabulary of dipeptides
    vectorizer = CountVectorizer(lowercase=False, ngram_range = (2,2), analyzer='char', vocabulary=voc, dtype = np.int8)#Initializes the object
    X = vectorizer.fit_transform(seqs).todense()#Creates count matrix of dipeptides
    dipeptides = pd.DataFrame(X)
    dipeptides['Charge'] = df['Charge']
    
    if save:
        #dipeptides.to_pickle(prefix_data+'dipeptide_fig4.pkl')
        np.save(prefix_data+'dipeptide_fig1', dipeptides.values, allow_pickle=True)

    return dipeptides

########## Tripeptides ############
def tripeptides_seq(save = False):
    """Read-in raw data, create vocabulary of tripeptides, fit a CountVectorizer object to get count of dipeptides"""
    prefix_data = paths.get_paths()['data']
    df = pd.read_csv(prefix_data+'SourceData_Figure_4.csv').loc[:,['Modified_sequence', 'Charge']]
    seqs = df['Modified_sequence'].str.replace('_','')
    aa = ['(', ')', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M','N', 'P', 'Q', 'R', 'S', 
    'T', 'V', 'W', 'Y', 'a', 'c', 'o','x']
    voc = [l3+l2+l1 for l1 in aa for l2 in aa for l3 in aa]#Creates vocabulary of tripeptides
    vectorizer = CountVectorizer(lowercase=False, ngram_range = (3,3), analyzer='char', vocabulary=voc, dtype = np.int8)#Init
    X = vectorizer.fit_transform(seqs).todense()#Creates count matrix of tripeptides
    tripeptides = pd.DataFrame(X, columns = voc)
    tripeptides['Charge'] = df['Charge']#Adds charge

    if save:
        prefix_data = '/mnt/pool-cox-data08/Juan/ccs/Data/'
        np.save(prefix_data+'tripeptide_fig4', tripeptides.values, allow_pickle=True)


######### Period 4 ##########
def helix_seq(save = False, helix_dip = True):
    """Read-in raw data, creates 5-pep vocab, creates count matrix and then sums up the terms correspondent to the same start/end character"""
    prefix_data = paths.get_paths()['data']
    df = pd.read_csv(prefix_data+'SourceData_Figure_1.csv').loc[:,['Modified sequence', 'Charge']]
    seqs = df['Modified sequence'].str.replace('_','')
    aa = ['(', ')', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M','N', 'P', 'Q', 'R', 'S', 
    'T', 'V', 'W', 'Y', 'a', 'c', 'o','x']
    voc = [l1+l3+l4+l5+l2 for l1 in aa for l2 in aa for l3 in aa for l4 in aa for l5 in aa]#5-pep vocab
    vectorizer = CountVectorizer(lowercase=False, ngram_range = (5,5), analyzer='char', vocabulary=voc, dtype = np.int8)
    X = vectorizer.fit_transform(seqs)#5-pep count matrix
    period4 = np.zeros((X.shape[0], 26*26), dtype = np.int8)
    for col in range(26*26):
        period4[:,col] = X[:,col*26*26*26:(col+1)*26*26*26].sum(1).A1#collapse it to 2-pep matrix (helix-like)
    #After manually inspecting the sequence, it looks fine

    if save:
        np.save(prefix_data+'period1_fig1', period4, allow_pickle=True)
    
    if helix_dip:# Merge helix and di-peptides
        dipeptides = np.load(prefix_data+'dipeptide_fig1.npy', allow_pickle = True)
        dip_hel = np.append(period4, dipeptides, axis = 1)#appends helix and dipeptides
        if save:
            np.save(prefix_data+'dip_hel_fig1.npy', dip_hel, allow_pickle=True)

# %%
