
#%%
import pandas as pd
import numpy as np
# %%
fig4 = pd.read_pickle('../Data/Fig1_powerlaw.pkl')
# %%
fig4_unmod = fig4[fig4['Modified sequence'].str.find('(')==-1]
fig4_unmod.loc[:,'Modified sequence'] = fig4_unmod['Modified sequence'].str.replace('_','')
fig4_ch2 = fig4_unmod.loc[fig4_unmod['Charge']==2,:]
shuffled = fig4_ch2.sample(frac = 1.0, random_state = 8112021)
shuffled['Modified sequence']
# %%
f = open('seqs.fasta', 'w')
for index, seq in shuffled['Modified sequence'].iteritems():
    f.write(f'>seqs\n{seq}\n')
f.close()
# %%
