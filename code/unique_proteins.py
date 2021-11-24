#%%
import numpy as np
import pandas as pd
from typing import Iterable

folder_path = '../Data/MaxQuant/CAEEL/'
file_path = 'peptides.txt'

df_peptides = pd.read_csv(folder_path+file_path, sep='\t')#Read data in

df_peptides_filtered = df_peptides[df_peptides['Reverse']!='+']#Filter out reversed peptides

print(df_peptides_filtered['Proteins'])
#%%
proteins_list_nested = df_peptides_filtered['Proteins'].apply(lambda x: x.split(';'))

print(proteins_list_nested)

# %%
def flatten(items):#flattens list
     """Yield items from any nested iterable."""
     for x in items:
         if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
             for sub_x in flatten(x):
                 yield sub_x
         else:
             yield x
#%%
proteins_list = list(flatten(proteins_list_nested.values.tolist()))
np.save(folder_path+'protein_list', proteins_list)
# %%
