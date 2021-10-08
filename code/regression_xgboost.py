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

# %%
############### Load in data and calculate error ####################
fig1 = pd.read_pickle('../Data/Fig1_powerlaw.pkl')
features_complete = np.load('../Data/one_hot_encoded_fig1.npy')
label_complete = (fig1['CCS'] - fig1['predicted_ccs']).values

charge = 3

features_ch2 = features_complete[features_complete[:,-1] == charge]
label_ch2 = label_complete[features_complete[:,-1] == charge]

#subsample
idx = np.random.choice(features_ch2.shape[0], 100000, replace = False)
features = features_ch2[idx]
label = label_ch2[idx]
#train/test set
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(features, label, test_size = 0.1, random_state=42)
print(f"The Initial Mean Absolute Error is: {sk.metrics.mean_absolute_error(fig1['CCS'], fig1['predicted_ccs'])}")



# %%
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
    mae = sk.metrics.mean_absolute_error(y_test, y_pred)
    print ("Mean Absolute Error: ", mae)
    
    return {'loss': mae, 'status': STATUS_OK, 'model': regressor}

def optimize(n_trials = 20, save_model = False):

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
        joblib.dump(xgb_best_bay, 'best_xgb_ch3')#Saves the best model
    
    return xgb_best_bay





# %%
############# Load-in/find optimal model #################
#xgb_best_bay = joblib.load('best_xgb')
xgb_best_bay = optimize(n_trials=4, save_model=True)
pred = xgb_best_bay.predict(x_test)
print(f"The Mean Absolute Error is: {sk.metrics.mean_absolute_error(y_test, pred)}")#Print error of best model




# %%
############ Test Set Results ###############
df_fig4 = pd.read_pickle('../Data/Fig4_powerlaw.pkl')
features_fig4 = np.load('../Data/one_hot_encoded_fig4.npy')
#%%

##### Separated by charge
charge = 2
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
# %%

##### Complete test set
df_fig4['xgboost'] = xgb_best_bay.predict(features_fig4) + df_fig4['predicted_ccs']

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 6))
ax[0].hist((df_fig4['CCS']-df_fig4['xgboost'])/df_fig4['xgboost']*100, bins = 50)
ax[0].set_xlabel('Relative Error of total prediction')
ax[0].set_ylabel('Counts')
ax[0].set_title('Relative error of residual w.r.t Ground Truth')

corr, _ = scipy.stats.pearsonr(df_fig4['xgboost'],df_fig4['CCS'])
ax[1].scatter(df_fig4['CCS'], df_fig4['xgboost'], label = f'Corr : {np.round(corr, 3)}', s = 0.1)
ax[1].set_xlabel('CCS')
ax[1].set_ylabel('Predicted CCS')
ax[1].set_title('Scatter Plot CCS vs predicted CCS')
ax[1].plot(np.arange(300,800), np.arange(300,800), 'b--')
ax[1].legend()

# %%