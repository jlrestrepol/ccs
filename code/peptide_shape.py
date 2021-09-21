'''
This module reads in MaxQuant's peptides output, then matches peptide sequence to a protein and then parses its
correspondent .pdb file and extracts the distance matrix for it.
'''

#%%
import numpy as np
import pandas as pd
import Bio.PDB
from distance_matrix import calc_dist_matrix_numpy


#Set paths for reading MaxQuant output
folder_path = '../Data/MaxQuant/CAEEL/'
file_path = 'peptides.txt'

df_peptides = pd.read_csv(folder_path+file_path, sep='\t')#Read data in

df_peptides_filtered = df_peptides[df_peptides['Reverse']!='+']#Filter out reversed peptides

df_pep_prot = df_peptides_filtered.loc[:,['Sequence','Proteins', 'Mass', 'Charges']]#Keep only useful features
# %%
#Set paths for reading Alpha Fold 2 output from
folder_path = '../Data/AlphaFold/CAEEL/AF-'
suffix = '-F1-model_v1.pdb'

#Get the data from the .pdb file
parser = Bio.PDB.PDBParser()# create parser
structure_alpha = parser.get_structure("alpha",folder_path+df_pep_prot['Proteins'][0]+suffix)#get structure
model_alpha = structure_alpha[0]#get model

#Calculate distance matrix
dist_matrix_alpha = calc_dist_matrix_numpy(model_alpha["A"], model_alpha["A"])
#%%
#Get sequences from .pdb file(proteins) and from MaxQuant's output(peptide)
ppb = Bio.PDB.PPBuilder().build_peptides(structure_alpha)[0]
seq_prot = ppb.get_sequence()
seq_pep = df_pep_prot['Sequence'][0]
#Find position of peptide in protein
start_index = seq_prot.find(seq_pep)
final_index = start_index + len(seq_pep)
# %%
#Extract submatrix
submatrix = dist_matrix_alpha[start_index: final_index, start_index:final_index]
# %%
