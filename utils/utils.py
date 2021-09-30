#%%
import pandas as pd
import numpy as np
from pyteomics import mass, parser
import numba  as nb
import re

# %%
df = pd.read_csv('../dl_paper/SourceData_Figure_1.csv')
df4 = pd.read_csv('../dl_paper/SourceData_Figure_4.csv')
df_sup = pd.read_csv('../dl_paper/Supplementary_Data_1.csv')
#%%
db = mass.Unimod()
aa_comp = dict(mass.std_aa_comp)
aa_comp['C'] += db.by_title('Carbamidomethyl')['composition']
aa_comp['(ac)'] = db.by_title('Acetyl')['composition']
aa_comp['(ox)'] = db.by_title('Oxidation')['composition']



# %%
unmodified_seq = df.loc[df['Modified sequence'].apply(lambda x: x.find('(') == -1), :]
unmodified_seq.loc[:,'Modified sequence'] = unmodified_seq['Modified sequence'].apply(lambda x: x[1:-1])

modified_seq = df.loc[df['Modified sequence'].apply(lambda x: x.find('(') != -1), :]
modified_seq.loc[:,'Modified sequence'] = modified_seq['Modified sequence'].apply(lambda x: x[1:-1])
#%%
mz_pred_mod = unmodified_seq.apply(lambda x: mass.calculate_mass(sequence = x['Modified sequence'], charge = x['Charge'], aa_comp=aa_comp), axis = 1)

# %%
#aa2formula = nb.typed.Dic.empty(key_values = nb.core.types.)
aa2formula = {
    'A': {'C': 3, 'H': 7, 'N': 1, 'O': 2},
    'R': {'C': 6, 'H': 14, 'N': 4, 'O': 2},
    'N': {'C': 4, 'H': 8, 'N': 2, 'O': 3},
    'D': {'C': 4, 'H': 7, 'N': 1, 'O': 4},
    'C': {'C': 3, 'H': 7, 'N': 1, 'O': 2, 'S': 1},
    '(ac)': {'C': 2, 'H': 2, 'O': 1},
    'Q': {'C': 5, 'H': 10, 'N': 2, 'O': 3},
    'E': {'C': 5, 'H': 9, 'N': 1, 'O': 4},
    'G': {'C': 2, 'H': 5, 'N': 1, 'O': 2},
    'H': {'C': 6, 'H': 9, 'N': 3, 'O': 2},
    'I': {'C': 6, 'H': 13, 'N': 1, 'O': 2},
    'L': {'C': 6, 'H': 13, 'N': 1, 'O': 2},
    'K': {'C': 6, 'H': 14, 'N': 2, 'O': 2},
    'M': {'C': 5, 'H': 11, 'N': 1, 'O': 2, 'S': 1},
    '(ox)': {'O': 1},
    'F': {'C': 9, 'H': 11, 'N': 1, 'O': 2},
    'P': {'C': 5, 'H': 9, 'N': 1, 'O': 2},
    'S': {'C': 3, 'H': 7, 'N': 1, 'O': 3},
    'T': {'C': 4, 'H': 9, 'N': 1, 'O': 3},
    'W': {'C': 11, 'H': 12, 'N': 2, 'O': 2},
    'Y': {'C': 9, 'H': 11, 'N': 1, 'O': 3},
    'V': {'C': 5, 'H': 11, 'N': 1, 'O': 2},
    'H-': {'H': 1},
    '-OH': {'O': 1, 'H': 1}
}
atom2mass = {
    'C': 12.0,
    'H': 1.00782503223,
    'O': 15.99491461956,
    'N': 14.00307400486,
    'S': 31.97207100
}
aa2formula['C'] = {'C': 5, 'H': 10, 'N': 2, 'O': 3, 'S': 1}

# %%
def calculate_mass(seq):
    mass = 0
    for aa in re.findall(r'\W..\W|[A-Z][a-z]?', seq):
        for atom, number_atoms in aa2formula[aa].items():
            mass += atom2mass[atom]*number_atoms
    aa_number = len(re.findall(r'[A-Z][a-z]?', seq))
    mass -= (aa_number-1)*(2*atom2mass['H'] + atom2mass['O'])
    return mass 

# %%
mass_calc = modified_seq.loc[:,'Modified sequence'].apply(calculate_mass)
mask = np.isclose(modified_seq.loc[:,'Mass'], mass_calc, atol = 1e-1)
modified_seq.loc[:, ['Modified sequence', 'Mass']][~mask]
# %%
mass_calc[~mask]

# %%
