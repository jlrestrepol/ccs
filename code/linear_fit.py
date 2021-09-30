#%%
import numpy as np
import pandas as pd
import Bio.PDB
from itertools import chain
from skspatial.objects import Line, Points
import matplotlib.pyplot as plt

# %%
data_folder = '/fs/pool/pool-cox-projects-fold/predictions/full_dbs/fasta/200/'
structure_alpha = Bio.PDB.PDBParser().get_structure("alpha",data_folder+"ranked_1.pdb")
# %%
model_alpha = structure_alpha[0]
chain_a = model_alpha['A']
# %%
coords_one = np.fromiter( chain.from_iterable(res["CA"].coord for res in chain_a), dtype = 'f8', count = -1).reshape((-1, 3))
points = Points(coords_one)
# %%
line_fit = Line.best_fit(points)

# %%
fig = plt.figure(figsize = (12,18))
ax=fig.add_subplot(211,projection='3d')
ax.view_init(30, 15)
line_fit.plot_3d(ax, t_1=-7, t_2=7, c='k')
points.plot_3d(ax, c='b', depthshade=False)
# %%
residuals = np.fromiter((line_fit.distance_point(point) for point in coords_one), dtype = 'f8', count = -1)
# %%
rmse = np.sqrt((residuals*residuals).sum()/coords_one.shape[0])
rmse
# %%
