#%%
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.svm import SVR, LinearSVR
from sklearn import model_selection
from sklearn.ensemble import BaggingRegressor
from sklearnex import patch_sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
patch_sklearn()
import joblib
import paths
# %%
def get_names(name):
    """Returns file name of tuple model file and data file"""

    model_name = {'Counts':'svr_count', 'Di-peptides':'svr_dip',
    'Extended':'svr_extended', 'Dip-hel':'svr_dip_hel'}
    data_name = {'One hot encoded':'one_hot_encoded_fig1.npy',
    'Counts':'counts_fig1.npy', 'Di-peptides':'dipeptide_fig1.npy', 
    'Tri-peptides' : 'tripeptides_fig1.npy',
    'Extended':'extended_fig1.npy', 'Dip-hel':'dip_hel_fig1.npy'}
    test_data_name = {'One hot encoded':'one_hot_encoded_fig4.npy',
    'Counts':'counts_fig4.npy', 'Di-peptides':'dipeptide_fig4.npy', 
    'Tri-peptides' : 'tripeptides_fig4.npy',
    'Extended':'extended_fig4.npy', 'Dip-hel':'dip_hel_fig4.npy'}
    return model_name[name], data_name[name], test_data_name[name]
#%%
name = 'Di-peptides'
_, data_name, _ = get_names(name)
prefix_data = paths.get_paths()['data']
fig1 = pd.read_pickle(prefix_data+'Fig1_powerlaw.pkl')
Lysn = fig1[fig1['Experiment'].str.find('LysN') != -1]
pp1 = Lysn[Lysn['Charge'] == 2]
print(prefix_data+data_name)
features_complete =  np.load(prefix_data+data_name, allow_pickle=True)
features_pp1 = features_complete[pp1.index]
label = (pp1['CCS'] - pp1['predicted_ccs']).values
# %%
n_estimators = 20
regr = BaggingRegressor(base_estimator=LinearSVR(dual =False, loss='squared_epsilon_insensitive'), n_estimators=n_estimators, 
random_state=0, n_jobs=-1, max_samples= 1.0/n_estimators, verbose = 1)
regr.fit(features_pp1, label)
# %%
_, _, data_test = get_names(name)
prefix_data = paths.get_paths()['data']

df_fig4 = pd.read_pickle(prefix_data+'/Fig4_powerlaw.pkl').reset_index()
features_fig4 = np.load(prefix_data+data_test)
# %%
Lysn_test = df_fig4[df_fig4['Modified_sequence'].apply(lambda x : (x[1] == 'K') or (x[4:6] == ')K'))]
pp1_test = Lysn_test[Lysn_test['Charge']==2]
# %%
features_pp1_test = features_fig4[pp1_test.index]
# %%
preds = regr.predict(features_pp1_test)
# %%
preds
# %%
pred_ccs = pp1_test['predicted_ccs']+preds
# %%
# %%
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (20, 6))
res_rel = (pp1_test['CCS'] - pred_ccs)/pred_ccs*100
ax.hist(res_rel, bins = 50, label = f'MAD = {np.round(scipy.stats.median_abs_deviation(res_rel), 4)}')
ax.set_xlabel('Relative Error of residual')
ax.set_ylabel('Counts')
ax.set_title('Relative error of residual w.r.t Ground Truth - Charge 2')
ax.legend()
# %%
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (20, 6))
res_rel = (pp1_test['CCS'] - pred_ccs)/pred_ccs*100
res_rel = res_rel[np.abs(res_rel) < 10]
ax.hist(res_rel, bins = 50, label = f'MAD = {np.round(scipy.stats.median_abs_deviation(res_rel), 4)}')
ax.set_xlabel('Relative Error of residual')
ax.set_ylabel('Counts')
ax.set_title('Relative error of residual w.r.t Ground Truth - Charge 2')
ax.legend()
# %%
