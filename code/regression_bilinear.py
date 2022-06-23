#%%
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import scipy
from sklearn.linear_model import LinearRegression

pd.options.mode.chained_assignment = None
#%%
class BilinearRegression():
    def  __init__(self) -> None:
        self.f_2 = lambda x,y: None
        self.f_3 = lambda x,y: None
        self.f_4 = lambda x,y: None

    
    def fit(self, df):
        if 'Charge' not in df:
            raise Exception('df doesnt contain charge')
        if 'CCS' not in df:
            raise Exception('df doesnt conatin CCS')
        if 'm/z' not in df:
            raise Exception('df doesnt contain m/z')
        
        #separate in charges
        df_ch2 = df[df['Charge']==2]
        df_ch3 = df[df['Charge']==3]
        df_ch4 = df[df['Charge']==4]

        self.f_2 = LinearRegression()
        self.f_3 = LinearRegression()
        self.f_4 = LinearRegression()

        self.f_2.fit(df_ch2[['m/z', 'Retention time']].values, df_ch2['CCS'].values)
        self.f_3.fit(df_ch3[['m/z', 'Retention time']].values, df_ch3['CCS'].values)
        self.f_4.fit(df_ch4[['m/z', 'Retention time']].values, df_ch4['CCS'].values)

    def predict(self, df, inplace = True):
        if 'Charge' not in df:
            raise Exception('df doesnt contain charge')
        if 'm/z' not in df:
            raise Exception('df doesnt contain m/z')
        if self.f_2.predict([[0,0]]) is None:
            raise Exception('f_2 is not trained')
        if self.f_3.predict([[0,0]]) is None:
            raise Exception('f_3 is not trained')
        if self.f_4.predict([[0,0]]) is None:
            raise Exception('f_4 is not trained')

        #predict
        df.loc[:,'bilinear_ccs'] = 0
        df.loc[df['Charge']==2,'bilinear_ccs'] = self.f_2.predict(df[df['Charge']==2][['m/z','Retention time']].values)
        df.loc[df['Charge']==3,'bilinear_ccs'] = self.f_3.predict(df[df['Charge']==3][['m/z','Retention time']].values)
        df.loc[df['Charge']==4,'bilinear_ccs'] = self.f_4.predict(df[df['Charge']==4][['m/z','Retention time']].values)
        if not inplace:
            linear_ccs = df['bilinear_ccs']
            del df['bilinear_ccs']
            return linear_ccs
    
    def error_fit_plot(self, df):
        if 'Charge' not in df:
            raise Exception('df doesnt contain charge')
        if 'm/z' not in df:
            raise Exception('df doesnt contain m/z')
        if self.f_2.predict([[0,0]]) is None:
            raise Exception('f_2 is not trained')
        if self.f_3.predict([[0,0]]) is None:
            raise Exception('f_3 is not trained')
        if self.f_4.predict([[0,0]]) is None:
            raise Exception('f_4 is not trained')
        
        x_plot = np.linspace(250, 1700, 200)
        fig = plt.gcf()
        fig.set_size_inches((16, 8))
        scatter = plt.scatter(df['m/z'], df['CCS'], c = df['Charge'], s = 0.01)
        power2, = plt.plot(x_plot, lr.f_2(x_plot) , 'b--', label = 'fit charge 2')
        power3, = plt.plot(x_plot, lr.f_3(x_plot) , 'b--', label = 'fit charge 3')
        power4, = plt.plot(x_plot, lr.f_4(x_plot) , 'b--', label = 'fit charge 4')
        plt.xlabel('m/z')
        plt.ylabel(r'CCA ($A^2$)')
        plt.title('Scatter plot: CCS vs m/z')
        legend1 = plt.legend(*scatter.legend_elements(), title = 'Charges')
        legend2 = plt.legend([power2, power3, power4], 
        [f'{np.round(popt[0], 2)}*x + {np.round(popt[1], 2)}' for popt in [lr.args_f2, lr.args_f3, lr.args_f4]], loc = 4)
        plt.gca().add_artist(legend1)

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
        i = 0
        for ax, df_it, f_i in zip(ax, [df[df['Charge']==2], df[df['Charge']==3], df[df['Charge']==4]],[lr.f_2, lr.f_3, lr.f_4]):
            sns.histplot((df_it['CCS']-f_i(df_it['m/z']))/f_i(df_it['m/z'])*100, ax = ax)
            ax.set_xlabel('Residual %')
            ax.set_ylabel('Count')
            ax.set_title(f'Charge {i+2}')
            i += 1
    
    def test_set_plot(self, df):
        if 'Charge' not in df:
            raise Exception('df doesnt contain charge')
        if 'm/z' not in df:
            raise Exception('df doesnt contain m/z')
        if self.f_2.predict([[0,0]]) is None:
            raise Exception('f_2 is not trained')
        if self.f_3.predict([[0,0]]) is None:
            raise Exception('f_3 is not trained')
        if self.f_4.predict([[0,0]]) is None:
            raise Exception('f_4 is not trained')

        self.predict(df)

        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 6))
        fig.suptitle('bilinear_ccs', fontsize = 12)
        res_rel = (df['CCS']-df['bilinear_ccs'])/df['bilinear_ccs']*100
        ax[0].hist(res_rel, bins = 50, label = f'MAD = {np.round(scipy.stats.median_abs_deviation(res_rel), 4)}')
        ax[0].set_xlabel('Relative Error of CCS')
        ax[0].set_ylabel('Counts')
        ax[0].set_title('Relative error of CCS w.r.t Ground Truth')
        ax[0].legend()

        corr, _ = scipy.stats.pearsonr(df['bilinear_ccs'],df['CCS'])
        ax[1].scatter(df['CCS'], df['bilinear_ccs'], label = f'Corr : {np.round(corr, 4)}', s = 0.1)
        ax[1].set_xlabel('CCS')
        ax[1].set_ylabel('Predicted CCS')
        ax[1].set_title('Scatter Plot CCS vs predicted CCS')
        ax[1].plot(np.arange(300,800), np.arange(300,800), 'b--')
        ax[1].legend()

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
        i = 0
        for ax, df_it in zip(ax, [df[df['Charge']==2], df[df['Charge']==3], df[df['Charge']==4]]):
            res_rel = (df_it['CCS']-df_it['bilinear_ccs'])/df_it['bilinear_ccs']*100
            sns.histplot(res_rel, ax = ax, label = f'MAD = {np.round(scipy.stats.median_abs_deviation(res_rel), 4)}')
            ax.set_xlabel('Residual %')
            ax.set_ylabel('Count')
            ax.set_title(f'Charge {i+2}')
            ax.legend()
            i += 1

#%%
if __name__ == "__main__":
    # Load data in
    df = pd.read_csv('../dl_paper/SourceData_Figure_1.csv')
    df4 = pd.read_csv('../dl_paper/SourceData_Figure_4.csv')
    df4['m/z'] = df4.apply(lambda x: utils.calculate_mass(x['Modified_sequence'], x['Charge']), axis = 1)
    lr = BilinearRegression()
    lr.fit(df)
    lr.predict(df4)  
    lr.error_fit_plot(df4)
    lr.test_set_plot(df4)

# %%
