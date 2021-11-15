#%%
import numpy as np
import pandas as pd
import torch
import paths
from sklearn import model_selection
import sklearn as sk
from transformers import TrainingArguments, Trainer
from transformers import AdamW, BertTokenizer, AutoModel, AutoModelForSequenceClassification
# %%
global path
path = paths.get_paths()
#%%
def train_val_set(charge = np.nan):
    """Function that outputs train and validation set for a given charge state"""
    fig1 = pd.read_pickle(path['data']+'Fig1_powerlaw.pkl')#loads in raw training data
    fig1 = fig1[fig1['Modified sequence'].str.find('(')==-1]#Unmodified seqs
    fig1.loc[:,'Modified sequence'] = fig1['Modified sequence'].str.replace('_','')
    fig1.loc[:,'Modified sequence'] = fig1['Modified sequence'].str.replace('', ' ')
    fig1.loc[:,'Modified sequence'] = fig1['Modified sequence'].apply(lambda x : x[1:-1])
    label_complete = (fig1['CCS'] - fig1['predicted_ccs']).values#residual
    features = fig1[fig1['Charge'] == charge]['Modified sequence']#choose points with given charge, drop charge feature because of 2 heads
    label = label_complete[fig1['Charge'] == charge]#choose appropiate residuals
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, label, test_size = 0.1, random_state=42)#train/val set split
    print(f"The Initial Mean Squared Error is: {sk.metrics.mean_squared_error(fig1['CCS'], fig1['predicted_ccs'])}")#prints initial error
    return x_train, x_test, y_train, y_test


#%%
training_args = TrainingArguments("test-trainer")
checkpoint = "Rostlab/prot_bert"
tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=False )
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)
# %%
x_train, x_test, y_train, y_test = train_val_set(2)
batch_train = tokenizer(x_train.values.tolist(), padding=True, return_tensors='pt')
batch_train["labels"] = y_train.tolist()
batch_test = tokenizer(x_test.values.tolist(), padding=True, return_tensors='pt')
batch_test["labels"] = y_test.tolist()
#%%
trainer = Trainer(
    model,
    training_args,
    train_dataset=batch_train,
    eval_dataset=batch_test,
    tokenizer=tokenizer,
)
# %%
trainer.train()
# %%
