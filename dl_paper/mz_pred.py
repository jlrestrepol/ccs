#%%
import sys
import numpy as np
import pandas as pd
sys.path.append('../utils/')

import utils
# %%
#The data was downloaded from https://www.ebi.ac.uk/pride/archive/projects/PXD017703
df_evidence = pd.read_csv('./Data/evidence.txt', sep = '\t')
df_peptides = pd.read_csv('./Data/peptides.txt', sep = '\t')
# %%
mass_calc  = df_evidence['Modified sequence'].apply(utils.calculate_mass)
mass_pred = df_evidence['Mass']
# %%
print(f"The calulation and the prediction are equal within a tolerance? {np.allclose(mass_calc, mass_pred)}")
# %%
df_fig4 = pd.read_csv('SourceData_Figure_4.csv', index_col=0)
# %%
Mod_seq = df_fig4['Modified_sequence'].apply(lambda x: x[1:-1])
# %%
df_peptides[df_peptides.apply(lambda x: x['Sequence'].find(')') != -1, axis = 1)]
# %%
inters = set(df_evidence['Modified sequence']).intersection(set(df_fig4['Modified_sequence']))
# %%
inters
# %%
