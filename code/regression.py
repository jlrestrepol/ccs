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
fig1 = pd.read_pickle('../Data/Fig1_powerlaw.pkl')
features = np.load('../Data/encoded_fig1.npy')
# %%
label = (fig1['CCS'] - fig1['predicted_ccs']).values
#%%
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(features, label, test_size = 0.1, random_state=42)
# %%
model = xgboost.XGBRegressor()
# %%
space = {
    'max_depth' : hp.choice('max_depth', range(5, 30, 1)),
    'learning_rate' : hp.quniform('learning_rate', 0.01, 0.5, 0.01),
    'n_estimators' : hp.choice('n_estimators', range(20, 205, 5)),
    'gamma' : hp.quniform('gamma', 0, 0.50, 0.01),
    'min_child_weight' : hp.quniform('min_child_weight', 1, 10, 1),
    'subsample' : hp.quniform('subsample', 0.1, 1, 0.01),
    'colsample_bytree' : hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)}
# %%
def objective(space):

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
# %%
tpe_algo = tpe.suggest
tpe_trials = Trials()
# %%
# Run 100 evals with the tpe algorithm
tpe_best = fmin(fn=objective, space=space, 
                algo=tpe_algo, trials=tpe_trials, 
                max_evals=20)
#%%
xgb_best_bay = xgboost.XGBRegressor(**tpe_best)                
xgb_best_bay.fit(x_train, y_train)
#%%
joblib.dump(xgb_best_bay, 'best_xgb')
#%%
df_fig4 = pd.read_pickle('../Data/Fig4_powerlaw.pkl')
features_fig4 = np.load('../Data/encoded_fig4.npy')
# %%
residual = xgb_best_bay.predict(x_test)
# %%
#pred = df_fig4['predicted_ccs'] + residual
pred = residual
# %%
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 6))
_ = sns.histplot((y_test- pred)/pred*100, ax = ax[0])
ax[0].set_xlabel('Relative Error')
ax[0].set_ylabel('Counts')
ax[0].set_title('Relative error w.r.t Ground Truth')

corr, _ = scipy.stats.pearsonr(y_test,pred)
ax[1].scatter(y_test, pred, label = f'Corr : {np.round(corr, 3)}', s = 0.1)
ax[1].set_xlabel('CCS')
ax[1].set_ylabel('Predicted CCS')
ax[1].set_title('Scatter Plot CCS predicted CCS')
ax[1].plot(np.arange(200,1200), np.arange(200,1200), 'b--')
ax[1].legend()
# %%
