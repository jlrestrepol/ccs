#%%
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import Ridge
from sklearn import model_selection
from sklearnex import patch_sklearn
import time
import matplotlib.pyplot as plt
import seaborn as sns

#%%
charge = 2
prefix_data = '/mnt/pool-cox-data08/Juan/ccs/Data/'
fig1 = pd.read_pickle(prefix_data+'Fig1_powerlaw.pkl')
features_complete =  np.load(prefix_data+'counts_fig1.npy', allow_pickle=True)
label_complete = (fig1['CCS'] - fig1['predicted_ccs']).values

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
# %%
start = time.time()
reg = Ridge()
reg.fit(x_train, y_train)
end = time.time()
print(end-start)
# %%
y_pred = reg.predict(x_test)
res = (y_pred-y_test)/y_pred
plt.hist(res[np.abs(res)<100])
#This looks horrible
# %%