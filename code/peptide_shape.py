'''
This module reads in MaxQuant's peptides output, then matches peptide sequence to a protein and then parses its
correspondent .pdb file and extracts the distance matrix for it.
'''

#%%
import numpy as np
import pandas as pd
import Bio.PDB
from distance_matrix import calc_dist_matrix_numpy
import os
import scipy

#Set paths for reading MaxQuant output
organism = 'CAEEL'
folder_path = '../Data/MaxQuant/'+organism+'/'
file_path = 'peptides.txt'

df_peptides = pd.read_csv(folder_path+file_path, sep='\t')#Read data in

df_peptides_filtered = df_peptides[df_peptides['Reverse']!='+']#Filter out reversed peptides

df_pep_prot = df_peptides_filtered.loc[:,['Sequence','Proteins', 'Mass', 'Charges']]#Keep only useful features

df_pep_prot['Proteins'] = df_pep_prot['Proteins'].apply(lambda x: x.split(';')[0])#Keep only the first protein - might have to change it
df_pep_prot['Proteins'] = df_pep_prot['Proteins'].apply(lambda x: x.split("-")[0])
df_pep_prot['Proteins'] = df_pep_prot['Proteins'].apply(lambda x: x.split("__")[-1])
#UNCOMMENT NEXT LINE IF ECOLI
#df_pep_prot['Proteins'] = df_pep_prot['Proteins'].apply(lambda x: x[x.find('|')+1:x.rfind('|')])

seq_list = df_pep_prot['Sequence'].values
#Set paths for reading Alpha Fold 2 output from
folder_path = '../Data/AlphaFold/'+organism+'/AF-'
suffix = '-F1-model_v1.pdb'
submatrix_list = []
count_no_match = 0


#%%
#Iterate over the protein names
for index, (label, prot_id) in enumerate(df_pep_prot['Proteins'].items()):
    #Get the data from the .pdb file
    
    file_path = folder_path+prot_id+suffix
    if not os.path.exists(file_path):
        count_no_match += 1
        print(seq_list[index], prot_id+' not found, total not found: '+str(count_no_match), 'index: '+str(index))
        submatrix_list.append(np.nan)
        continue
    #print(prot_id, seq_list[index], index, 'total not found '+ str(count_no_match))
    parser = Bio.PDB.PDBParser()# create parser
    structure_alpha = parser.get_structure("alpha",folder_path+prot_id+suffix)#get structure
    model_alpha = structure_alpha[0]#get model

    #Calculate distance matrix
    dist_matrix_alpha = calc_dist_matrix_numpy(model_alpha["A"], model_alpha["A"])

    #Get sequences from .pdb file(proteins) and from MaxQuant's output(peptide)
    ppb = Bio.PDB.PPBuilder().build_peptides(structure_alpha)[0]
    seq_prot = ppb.get_sequence()
    seq_pep = seq_list[index]
    #Find position of peptide in protein
    start_index = seq_prot.find(seq_pep)
    final_index = start_index + len(seq_pep)

    #Extract submatrix
    submatrix = dist_matrix_alpha[start_index: final_index, start_index:final_index]
    submatrix_list.append(scipy.spatial.distance.squareform(submatrix))
    '''if index == 100:
        break'''
print('Total not found :' + str(count_no_match))
df_pep_prot['Distance_matrix'] = submatrix_list
results_folder = "../Data/Combined/"
df_pep_prot.to_hdf(results_folder+organism+'.h5','df')
# %%
