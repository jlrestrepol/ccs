#%%
import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import sys
# %%
fig1 = pd.read_csv('../dl_paper/SourceData_Figure_1.csv')
fig4 = pd.read_csv('../dl_paper/SourceData_Figure_4.csv')
#%%
seqs1 = fig1['Modified sequence']
seqs4 = fig4['Modified_sequence']
seqs1_set =set(seqs1)
seqs4_set = set(seqs4)
common = seqs1_set.intersection(seqs4_set)
print(f' The number of common elements is: {len(common)}')
#%%
seqs1_unique = seqs1.unique()
seqs4_unique = seqs4.unique()
appended = np.append(seqs1_unique, seqs4_unique, axis = 0)
appended_unique = np.unique(appended)
print(f'Shape of array 1: {seqs1_unique.shape}, Shape of array 2: {seqs4_unique.shape}, '\
    f'Shape of the sum = {seqs1_unique.shape[0] + seqs4_unique.shape[0]}, '\
    f'unique sequences = {appended_unique.shape},'\
    f'Number of common sequences = {seqs1_unique.shape[0] + seqs4_unique.shape[0] - appended_unique.shape[0]} ')
# %%
