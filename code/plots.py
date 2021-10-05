#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dir_path = "/fs/pool/pool-cox-projects-fold/"
file_name = "Supplementary_Data_1.csv"

df = pd.read_csv(dir_path+file_name, index_col="id")

from pyteomics import mass, parser

#Calculate mass of peptide and save it in a new column
df["Mass"] = df["Modified_sequence"].apply(lambda x: mass.calculate_mass(parser.parse(x, show_unmodified_termini = True)))
df["m/z"] = df["Mass"] / df["Charge"]#Mass to charge ratio of peptide

file_mass = "Supplementary_Data_1_with_mass.csv"
df.set_index("id").to_csv(dir_path+file_mass)#Save file
#%%
df_mass = pd.read_csv(dir_path+file_mass, index_col="id")#Read in file
scatter = plt.scatter(df_mass['m/z'], df_mass['CCS_Prediction'], s = 0.1, c=df_mass['Charge'])
plt.xlabel("m/z")
plt.ylabel("CCS")
plt.legend(*scatter.legend_elements(), title = "Charge")
plt.title("Data points in CCS vs m/z space. Charge colored.")
# %%
