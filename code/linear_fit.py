#%%
import numpy as np
import pandas as pd
import Bio.PDB
from Bio.PDB.Polypeptide import PPBuilder
from itertools import chain
from skspatial.objects import Line, Points
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pickle

#%%

def one_fit(file_name = '/200/ranked_1.pdb'):
    data_folder = '/fs/pool/pool-cox-projects-fold/predictions/full_dbs/fasta'#Folder path
    structure_alpha = Bio.PDB.PDBParser().get_structure("alpha",data_folder+file_name)

    model_alpha = structure_alpha[0]
    chain_a = model_alpha['A']#Fetch chain
    coords_one = np.fromiter( chain.from_iterable(res["CA"].coord for res in chain_a), dtype = 'f8', count = -1).reshape((-1, 3))
    points = Points(coords_one)#Create Points object to plot easily
    line_fit = Line.best_fit(points)#Performs least square fit

    residuals = np.fromiter((line_fit.distance_point(point) for point in points), dtype = 'f8', count = -1)#Calculate residuals
    rmse = np.sqrt((residuals*residuals).sum()/coords_one.shape[0])#Calculate RMSE
    
    fig = plt.figure(figsize = (12,18))
    ax=fig.add_subplot(211,projection='3d')
    ax.view_init(30, 15)
    line_fit.plot_3d(ax, t_1=-7, t_2=7, c='k', label = 'linear fit')
    points.plot_3d(ax, c='b', depthshade=False, label = 'Amino acids')
    ax.set_xlabel('x(A)')
    ax.set_ylabel('y(A)')
    ax.set_zlabel('z(A)')
    ax.set_title('Linear Fit to amino acid sequence')
    ax.legend()
    
    return rmse

#%%
def predicted():
    data_folder = '/fs/pool/pool-cox-projects-fold/predictions/full_dbs/fasta'
    rmse_list = []
    seq_list = []
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
                    coords_one = np.fromiter( chain.from_iterable(res["CA"].coord for res in chain_a), dtype = 'f8', count = -1).reshape((-1, 3))
                    points = Points(coords_one)
                    line_fit = Line.best_fit(points)
                    residuals = np.fromiter((line_fit.distance_point(point) for point in points), dtype = 'f8', count = -1)
                    rmse = np.sqrt((residuals*residuals).sum()/coords_one.shape[0])
                    rmse_list.append(rmse)
    
    fig = plt.figure(figsize = (12,18))
    ax=fig.add_subplot(111)
    ax = sns.histplot(rmse_list)
    ax.set_xlabel('RMSE')
    ax.set_ylabel('Counts')
    ax.set_title('Distribution of RMSE-PLDDT>70')
    
    return rmse_list, seq_list

#%%
def downloaded():
    data_folder = '../Data/AlphaFold/'
    rmse_list = []
    seq_list = []
    for root,dirs,files in os.walk(data_folder):

        if len(dirs) > 0:
            continue
        print(root)
        for i, file in enumerate(files): 
            structure_alpha = Bio.PDB.PDBParser().get_structure("alpha",root+'/'+file)
            ppb=PPBuilder()
            seq = str([pp.get_sequence() for pp in ppb.build_peptides(structure_alpha)][0])
            seq_list.append(seq)
            model_alpha = structure_alpha[0]
            chain_a = model_alpha['A']
            coords_one = np.fromiter( chain.from_iterable(res["CA"].coord for res in chain_a), dtype = 'f8', count = -1).reshape((-1, 3))
            points = Points(coords_one)
            line_fit = Line.best_fit(points)
            residuals = np.fromiter((line_fit.distance_point(point) for point in points), dtype = 'f8', count = -1)
            rmse = np.sqrt((residuals*residuals).sum()/coords_one.shape[0])
            rmse_list.append(rmse)
            if i%500 == 0:
                print(f'{i} read out of {len(files)}')
        
    fig = plt.figure(figsize = (12,18))
    ax=fig.add_subplot(111)
    ax = sns.histplot(rmse_list)
    ax.set_xlabel('RMSE')
    ax.set_ylabel('Counts')
    ax.set_title('Distribution of RMSE-PLDDT>70')
    
    return rmse_list, seq_list
#%%
if __name__ == '__main__':
    rmse_list, seq_list = downloaded()
    
    with open("rmse_list", "wb") as fp:
        pickle.dump(rmse_list, fp)

    with open("seq_list", "wb") as fp:
        pickle.dump(seq_list, fp)
    
    '''
    rmse_array = np.array(rmse_list)
    seq_array = np.array(seq_list)
    second_peak = seq_array[(rmse_array>2.0) & (rmse_array<2.44)]

    df = pd.read_csv('../dl_paper/SourceData_Figure_1.csv')
    seqs_fig1 = df['Modified sequence'].str.replace('_','')
    set_af = set(seq_array)
    set_exp = set(seqs_fig1)
    inters = set_af.intersection(set_exp)
    df['Modified sequence'] = seqs_fig1
    df.set_index('Modified sequence', inplace = True)
    df_inters = df.loc[list(inters)]
    
    fig = plt.figure(figsize = (12,18))
    ax=fig.add_subplot(111)
    fig.set_size_inches((16, 8))
    scatter = plt.scatter(df['m/z'], df['CCS'], c = df['Charge'], s = 0.01)
    scatter2 = plt.scatter(df_inters['m/z'], df_inters['CCS'], c = 'green', s = 5)
    plt.xlabel('m/z')
    plt.ylabel(r'CCA ($A^2$)')
    plt.title('Scatter plot: CCA vs m/z')
    plt.legend(*scatter.legend_elements(), title = 'Charges')'''
# %%
