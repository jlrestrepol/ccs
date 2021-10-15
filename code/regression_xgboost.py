#%%
import pandas as pd
import numpy as np
import scipy
import xgboost
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import scipy
import scipy
#%%
############# Obtimization routine #######################
def objective(space):
    """Function that will be optimized"""
    regressor = xgboost.XGBRegressor(n_estimators = space['n_estimators'],
                            max_depth = int(space['max_depth']),
                            learning_rate = space['learning_rate'],
                            gamma = space['gamma'],
                            min_child_weight = space['min_child_weight'],
                            subsample = space['subsample'],
                            colsample_bytree = space['colsample_bytree']
                            )
    
    regressor.fit(x_train, y_train)
    
    y_pred = regressor.predict(x_test)
    mse = sk.metrics.mean_squared_error(y_test, y_pred)
    print ("Mean Squared Error: ", mse)
    
    return {'loss': mse, 'status': STATUS_OK, 'model': regressor}

def optimize(n_trials = 20, save_model = False, file_name = ""):

    """Runs Bayesian hyperparameter optimization on XGBoost"""
    space = {
    'max_depth' : hp.choice('max_depth', range(5, 30, 1)),
    'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
    'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
    'gamma' : hp.quniform('gamma', 0, 0.50, 0.01),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
    'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
    'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)}

    tpe_algo = tpe.suggest
    tpe_trials = Trials()

    # Run 100 evals with the tpe algorithm
    tpe_best = fmin(fn=objective, space=space, 
                    algo=tpe_algo, trials=tpe_trials, 
                    max_evals=n_trials)
    
    xgb_best_bay = xgboost.XGBRegressor(**tpe_best)#Instantiates best regressor
    xgb_best_bay.fit(x_train, y_train)#Fit best
    
    if save_model:
        joblib.dump(xgb_best_bay, file_name)#Saves the best model
    
    return xgb_best_bay


def test_set_one_charge_results(charge):
    '''Plots results on the test set'''
    df_fig4 = pd.read_pickle('../Data/Fig4_powerlaw.pkl')
    features_fig4 = np.load('../Data/one_hot_encoded_fig4.npy')
    
    xgb_best_bay = joblib.load(f'xgb_counts_ch{charge}')

    ##### Separated by charge
    res  = xgb_best_bay.predict(features_fig4[features_fig4[:,-1] == charge])
    pred_ccs = df_fig4[df_fig4['Charge']==charge]['predicted_ccs'] + res

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 6))
    ax[0].hist((df_fig4[df_fig4['Charge']==charge]['predicted_ccs']-pred_ccs)/pred_ccs*100, bins = 50)
    ax[0].set_xlabel('Relative Error of residual')
    ax[0].set_ylabel('Counts')
    ax[0].set_title('Relative error of residual w.r.t Ground Truth - Charge 2')

    corr, _ = scipy.stats.pearsonr(df_fig4[df_fig4['Charge']==charge]['predicted_ccs'],pred_ccs)
    ax[1].scatter(df_fig4[df_fig4['Charge']==charge]['predicted_ccs'], pred_ccs, label = f'Corr : {np.round(corr, 3)}', s = 0.1)
    ax[1].set_xlabel('Residual')
    ax[1].set_ylabel('Predicted Residual')
    ax[1].set_title('Scatter Plot Residual vs predicted Residual')
    ax[1].plot(np.arange(300,600), np.arange(300,600), 'b--')
    ax[1].legend(loc = 'lower right')

def test_set_results():
    '''Results on the complete test set'''
    prefix = '../models/xgboost_extended/xgb_extended'
    xgb_ch2 = joblib.load(prefix+'_ch2')
    xgb_ch3 = joblib.load(prefix+'_ch3')
    xgb_ch4 = joblib.load(prefix+'_ch4')

    df_fig4 = pd.read_pickle('../Data/Fig4_powerlaw.pkl')
    features_fig4 = np.load('../Data/extended_fig4.npy', allow_pickle=True)

    df_fig4['xgboost'] = 0
    df_fig4.loc[df_fig4['Charge']==2,'xgboost'] = xgb_ch2.predict(features_fig4[features_fig4[:,-1]==2]) + df_fig4.loc[df_fig4['Charge']==2,'predicted_ccs']
    df_fig4.loc[df_fig4['Charge']==3,'xgboost'] = xgb_ch3.predict(features_fig4[features_fig4[:,-1]==3]) + df_fig4.loc[df_fig4['Charge']==3,'predicted_ccs']
    df_fig4.loc[df_fig4['Charge']==4,'xgboost'] = xgb_ch4.predict(features_fig4[features_fig4[:,-1]==4]) + df_fig4.loc[df_fig4['Charge']==4,'predicted_ccs']

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 6))
    res_rel = (df_fig4['CCS']-df_fig4['xgboost'])/df_fig4['xgboost']*100
    ax[0].hist(res_rel, bins = 50, label = f'MAD = {np.round(scipy.stats.median_abs_deviation(res_rel), 4)}')
    ax[0].set_xlabel('Relative Error of CCS')
    ax[0].set_ylabel('Counts')
    ax[0].set_title('Relative error of CCS w.r.t Ground Truth')
    ax[0].legend()

    corr, _ = scipy.stats.pearsonr(df_fig4['xgboost'],df_fig4['CCS'])
    ax[1].scatter(df_fig4['CCS'], df_fig4['xgboost'], label = f'Corr : {np.round(corr, 4)}', s = 0.1)
    ax[1].set_xlabel('CCS')
    ax[1].set_ylabel('Predicted CCS')
    ax[1].set_title('Scatter Plot CCS vs predicted CCS')
    ax[1].plot(np.arange(300,800), np.arange(300,800), 'b--')
    ax[1].legend()


def bayessian_opt(charge):

    ############### Load in data and calculate error ####################
    fig1 = pd.read_pickle('../Data/Fig1_powerlaw.pkl')
    counts_aa = np.load('../Data/counts_fig1.npy', allow_pickle=True)
    counts_dip = pd.read_pickle('../Data/dipeptide_fig1.pkl').values
    features_complete = np.append(counts_aa[:,:-1], counts_dip, axis = 1)
    label_complete = (fig1['CCS'] - fig1['predicted_ccs']).values
    
    features_ch2 = features_complete[features_complete[:,-1] == charge]
    label_ch2 = label_complete[features_complete[:,-1] == charge]
    #subsample
    idx = np.random.choice(features_ch2.shape[0], features_ch2.shape[0], replace = False)
    features = features_ch2[idx]
    label = label_ch2[idx]

    del features_complete
    del label_complete
    del features_ch2
    del label_ch2
    #train/test set
    print(features.shape, label.shape)
    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(features, label, test_size = 0.1, random_state=42)
    del features
    del label
    print(f"The Initial Mean Squared Error is: {sk.metrics.mean_squared_error(fig1['CCS'], fig1['predicted_ccs'])}")
    del fig1
    ############# Load-in/find optimal model #################
    #xgb_best_bay = joblib.load('best_xgb')
    xgb_best_bay = optimize(n_trials=20, save_model=True, file_name=f'xgb_extended_ch{charge}')
    pred = xgb_best_bay.predict(x_test)
    print(f"The Mean Squared Error is: {sk.metrics.mean_squared_error(y_test, pred)}")#Print error of best model
# %%

if __name__ == "__main__":
    charge = 2
    bayessian_opt(charge)
    charge = 3
    bayessian_opt(charge)
    charge = 4
    bayessian_opt(charge)

