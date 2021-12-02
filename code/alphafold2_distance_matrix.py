'''
This module reads in MaxQuant's peptides output, then matches peptide sequence to a protein and then parses its
correspondent .pdb file and extracts the distance matrix for it.
'''

#%%
import numpy as np
import pandas as pd
import Bio.PDB
from Bio.PDB.Polypeptide import PPBuilder
from distance_matrix import calc_dist_matrix_numpy
import os
import scipy
import paths
import json
from linear_fit import predicted, scatter_plot
import pickle
import matplotlib.pyplot as plt
# %%
path = paths.get_paths()
pfx_fasta = path['af2-fasta']
chrg_dir = 'charge2/'
fasta_path = pfx_fasta+chrg_dir
pfx_pep = path['af2-data']
pep_path = pfx_pep+chrg_dir
# %%
rmse_list, seq_list = predicted(data_folder = pep_path)
#%%
with open("rmse_list.pkl", "wb") as fp:
    pickle.dump(rmse_list, fp)

with open("seq_list.pkl", "wb") as fp:
    pickle.dump(seq_list, fp)

#%%
rmse_list = pd.read_pickle('rmse_list.pkl')    
seq_list = pd.read_pickle('seq_list.pkl')
scatter_plot(rmse_list, seq_list)
#%%
def distance_matrix_list(data_folder = ''):
    dist_list = []
    seq_list = []
    i = 0
    print("Walking through dir")
    for root,dirs,files in os.walk(data_folder):
        #go through roots that don't end in fasta or msas and only take ones that are completed
        if not (root.endswith(("fasta","msas"))) and len(files)==23:
            f=os.path.join(root,"ranking_debug.json")
            with open(f) as json_data:
                data=json.load(json_data)
                model_scores=data['plddts']

                best_model=max(model_scores, key=model_scores.get)
                best_score=model_scores[best_model]

                if best_score > 70: #Only take files whose score is above 70
                    structure_alpha = Bio.PDB.PDBParser().get_structure("alpha",root+"/ranked_1.pdb")
                    ppb=PPBuilder()
                    seq = str([pp.get_sequence() for pp in ppb.build_peptides(structure_alpha)][0])
                    seq_list.append(seq)
                    model_alpha = structure_alpha[0]
                    chain_a = model_alpha['A']
                    dist_matrix = calc_dist_matrix_numpy(model_alpha["A"], model_alpha["A"])
                    dist_list.append(dist_matrix)
                    if i % 1000 == 0:
                        print(f'{i} structures with good confidence fitted')
                    i += 1
    return dist_list, seq_list
# %%
dist_list, seq_list = distance_matrix_list(data_folder = pep_path)
# %%
#Sanity check: shapes and seqs lengths are equal
lens = np.fromiter(map(len, seq_list), dtype = np.int32)
shapes = np.fromiter((e.shape[0] for e in dist_list), dtype = np.int32)
np.array_equal(lens, shapes)
# %%
len(dist_list)
# %%
#%%
with open("distance_list.pkl", "wb") as fp:
    pickle.dump(dist_list, fp)

with open("seq_dist_list.pkl", "wb") as fp:
    pickle.dump(seq_list, fp)
# %%
