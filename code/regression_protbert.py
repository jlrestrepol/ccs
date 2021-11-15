#%%
import numpy as np
from transformers import BertForMaskedLM, BertTokenizer, pipeline, AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
# %%
checkpoint = "Rostlab/prot_bert"
tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=False )
#model = BertForMaskedLM.from_pretrained(checkpoint)
# %%
inputs = tokenizer(['A A A C L P E P', 'A A A C C'], padding=True, return_tensors='tf')
# %%
inputs
# %%
