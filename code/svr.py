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
#%%
def train_test_set(charge, name, add_rt = False):
    _, data_name, _ = get_names(name)
    prefix_data = paths.get_paths()['data']
    fig1 = pd.read_pickle(prefix_data+'Fig1_powerlaw.pkl')
    print(prefix_data+data_name, add_rt)
    features_complete =  np.load(prefix_data+data_name, allow_pickle=True)
    label_complete = (fig1['CCS'] - fig1['predicted_ccs']).values
    if add_rt:
        features_complete = np.hstack((fig1['Retention time'].values[:,None], features_complete))
        mask = fig1['Retention time'].values < 120
        features_complete = features_complete[mask]
        label_complete = label_complete[mask]
    ss = StandardScaler()
    features_complete[:,:-1] = ss.fit_transform(features_complete[:,:-1])#Fit transform excluding charge

    print('Take specific charge state')
    features_ch2 = features_complete[features_complete[:,-1] == charge]
    label_ch2 = label_complete[features_complete[:,-1] == charge]
    #subsample
    print('Subsample')
    idx = np.random.choice(features_ch2.shape[0], features_ch2.shape[0], replace = False)
    features = features_ch2[idx]
    label = label_ch2[idx]

    del features_complete
    del label_complete
    del features_ch2
    del label_ch2
    #train/test set
    print(features.shape, label.shape)
    #global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, label, test_size = 0.1, random_state=42)
    return x_train, x_test, y_train, y_test
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
def test_set_results(name, add_rt = False):
    '''Results on the complete test set'''
    model_name, data_train, data_name = get_names(name)
    prefix_models = paths.get_paths()['models']
    #prefix_models = ''
    #model_name = '.'
    svr_ch2 = joblib.load(prefix_models+model_name+'/svr_ch2_rt')
    svr_ch3 = joblib.load(prefix_models+model_name+'/svr_ch3_rt')
    svr_ch4 = joblib.load(prefix_models+model_name+'/svr_ch4_rt')
    prefix_data = paths.get_paths()['data']
    df_fig4 = pd.read_pickle(prefix_data+'Fig4_with_RT.pkl')
    fig1 = pd.read_pickle(prefix_data+'Fig1_powerlaw.pkl')
    features_fig4 = np.load(prefix_data+data_name, allow_pickle=True)
    features_fig1 =  np.load(prefix_data+data_train, allow_pickle=True)
    if add_rt:
        features_fig4 = np.hstack((df_fig4['Retention time'].values[:,None], features_fig4))
        features_fig1 = np.hstack((fig1['Retention time'].values[:,None], features_fig1))
        mask4 = df_fig4['Retention time'].values < 120
        mask1 = fig1['Retention time'].values < 120
        features_fig1 = features_fig1[mask1]
        features_fig4 = features_fig4[mask4]

    ss = StandardScaler()
    ss.fit(features_fig1[:,:-1])#Fit transform excluding charge
    features_fig4[:,:-1] = ss.transform(features_fig4[:,:-1])#Fit transform excluding charge

    #features_fig4 = scaler.transform(features_fig4)
    print('Predicting')
    df_fig4['svr'] = 0
    df_fig4.loc[df_fig4['Charge']==2,'svr'] = svr_ch2.predict(features_fig4[features_fig4[:,-1]==2]) + df_fig4.loc[df_fig4['Charge']==2,'predicted_ccs']
    df_fig4.loc[df_fig4['Charge']==3,'svr'] = svr_ch3.predict(features_fig4[features_fig4[:,-1]==3]) + df_fig4.loc[df_fig4['Charge']==3,'predicted_ccs']
    df_fig4.loc[df_fig4['Charge']==4,'svr'] = svr_ch4.predict(features_fig4[features_fig4[:,-1]==4]) + df_fig4.loc[df_fig4['Charge']==4,'predicted_ccs']

    print('Creating histogram')
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 6))
    fig.suptitle(model_name, fontsize = 12)
    res_rel = (df_fig4['CCS']-df_fig4['svr'])/df_fig4['svr']*100
    ax[0].hist(res_rel, bins = 50, label = f'MAD = {np.round(scipy.stats.median_abs_deviation(res_rel), 4)}')
    ax[0].set_xlabel('Relative Error of CCS')
    ax[0].set_ylabel('Counts')
    ax[0].set_title('Relative error of CCS w.r.t Ground Truth')
    ax[0].legend()

    print('Creating scatter plot')
    corr, _ = scipy.stats.pearsonr(df_fig4['svr'],df_fig4['CCS'])
    ax[1].scatter(df_fig4['CCS'], df_fig4['svr'], label = f'Corr : {np.round(corr, 4)}', s = 0.1)
    ax[1].set_xlabel('CCS')
    ax[1].set_ylabel('Predicted CCS')
    ax[1].set_title('Scatter Plot CCS vs predicted CCS')
    ax[1].plot(np.arange(300,800), np.arange(300,800), 'b--')
    ax[1].legend()

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
    i = 0
    for ax, df_it in zip(ax, [df_fig4[df_fig4['Charge']==2], df_fig4[df_fig4['Charge']==3], df_fig4[df_fig4['Charge']==4]]):
        res_rel = (df_it['CCS']-df_it['svr'])/df_it['svr']*100
        sns.histplot(res_rel, ax = ax, label = f'MAD = {np.round(scipy.stats.median_abs_deviation(res_rel), 4)}')
        ax.set_xlabel('Residual %')
        ax.set_ylabel('Count')
        ax.set_title(f'Charge {i+2}')
        ax.legend()
        i += 1

    return df_fig4['svr']
# %%
def test_set_one_charge_results(model, charge, name):
    '''Plots results on the test set'''
    _, ddata_train, data = get_names(name)
    prefix_data = paths.get_paths()['data']

    df_fig4 = pd.read_pickle(prefix_data+'/Fig4_powerlaw.pkl')
    features_fig4 = np.load(prefix_data+data)
    #features_fig4 = scaler.transform(features_fig4)

    ##### Separated by charge
    print('Predicting')
    res  = model.predict(features_fig4[features_fig4[:,-1] == charge])
    pred_ccs = df_fig4[df_fig4['Charge']==charge]['predicted_ccs'] + res

    print('Creating histogram')
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 6))
    res_rel = (df_fig4[df_fig4['Charge']==charge]['predicted_ccs']-pred_ccs)/pred_ccs*100
    ax[0].hist(res_rel, bins = 50, label = f'MAD = {np.round(scipy.stats.median_abs_deviation(res_rel), 4)}')
    ax[0].set_xlabel('Relative Error of residual')
    ax[0].set_ylabel('Counts')
    ax[0].set_title('Relative error of residual w.r.t Ground Truth - Charge 2')
    ax[0].legend()

    print('Creating scatter plot')
    corr, _ = scipy.stats.pearsonr(df_fig4[df_fig4['Charge']==charge]['predicted_ccs'],pred_ccs)
    ax[1].scatter(df_fig4[df_fig4['Charge']==charge]['predicted_ccs'], pred_ccs, label = f'Corr : {np.round(corr, 3)}', s = 0.1)
    ax[1].set_xlabel('Residual')
    ax[1].set_ylabel('Predicted Residual')
    ax[1].set_title('Scatter Plot Residual vs predicted Residual')
    ax[1].plot(np.arange(300,600), np.arange(300,600), 'b--')
    ax[1].legend(loc = 'lower right')
#%%
def train(charge, name, save = False, n_estimators = 10, n_jobs = -1, add_rt=False):
    _, data, _ = get_names(name)
    x_train, x_test, y_train, y_test = train_test_set(charge, name, add_rt=add_rt)
    x_train = np.append(x_train, x_test, axis = 0)
    y_train = np.append(y_train, y_test, axis = 0)
    #ss = StandardScaler()
    #x_train = ss.fit_transform(x_train)
    print('Starting Training')
    start = time.time()
    regr = BaggingRegressor(base_estimator=LinearSVR(dual =False, loss='squared_epsilon_insensitive'), n_estimators=n_estimators, 
    random_state=0, n_jobs=n_jobs, max_samples= 1.0/n_estimators, verbose = 1)
    regr.fit(x_train, y_train)
    end = time.time()
    print(end-start)
    if save:
        prefix_models = paths.get_paths()['models']
        name_model = f'svr_ch{charge}_rt'
        joblib.dump(regr, prefix_models+'svr_dip/'+name_model)
    return regr
if __name__ == "__main__":
    # %%
    regr2 = train(2, 'Di-peptides', save = True)
    # %%
    regr3 = train(3, 'Di-peptides',save = True)
    # %%
    regr4 = train(4, 'Di-peptides',save = True)
    # %%
    test_set_results()
    # %%
