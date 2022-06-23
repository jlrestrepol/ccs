#%%
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import scipy
#%%
class LinearRegression():
    def  __init__(self) -> None:
        self.f_2 = lambda x: None
        self.f_3 = lambda x: None
        self.f_4 = lambda x: None
        self.args_f2 = None
        self.args_f3 = None
        self.args_f4 = None

    
    def fit(self, df, independent_variable = 'm/z'):
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

        f_lin = lambda x, a, b: a*x + b

        popt_2, _ =scipy.optimize.curve_fit(f_lin, df_ch2[independent_variable], df_ch2['CCS'])
        popt_3, _ =scipy.optimize.curve_fit(f_lin, df_ch3[independent_variable], df_ch3['CCS'])
        popt_4, _ =scipy.optimize.curve_fit(f_lin, df_ch4[independent_variable], df_ch4['CCS'])

        self.f_2 = lambda x: f_lin(x, a=popt_2[0], b=popt_2[1])
        self.f_3 = lambda x: f_lin(x, a=popt_3[0], b=popt_3[1])
        self.f_4 = lambda x: f_lin(x, a=popt_4[0], b=popt_4[1])

        self.args_f2 = [popt_2[0], popt_2[1]]
        self.args_f3 = [popt_3[0], popt_3[1]]
        self.args_f4 = [popt_4[0], popt_4[1]]
    
    def predict(self, df, inplace = True, independent_variable = 'm/z'):
        if 'Charge' not in df:
            raise Exception('df doesnt contain charge')
        if 'm/z' not in df:
            raise Exception('df doesnt contain m/z')
        if self.f_2(0) is None:
            raise Exception('f_2 is not trained')
        if self.f_3(0) is None:
            raise Exception('f_3 is not trained')
        if self.f_4(0) is None:
            raise Exception('f_4 is not trained')

        #predict
        df['linear_ccs'] = 0
        df.loc[df['Charge']==2,'linear_ccs'] = self.f_2(df[df['Charge']==2][independent_variable])
        df.loc[df['Charge']==3,'linear_ccs'] = self.f_3(df[df['Charge']==3][independent_variable])
        df.loc[df['Charge']==4,'linear_ccs'] = self.f_4(df[df['Charge']==4][independent_variable])
        if not inplace:
            linear_ccs = df['linear_ccs']
            del df['linear_ccs']
            return linear_ccs
    
    def error_fit_plot(self, df, independent_variable = 'm/z'):
        if 'Charge' not in df:
            raise Exception('df doesnt contain charge')
        if 'm/z' not in df:
            raise Exception('df doesnt contain m/z')
        if self.f_2(0) is None:
            raise Exception('f_2 is not trained')
        if self.f_3(0) is None:
            raise Exception('f_3 is not trained')
        if self.f_4(0) is None:
            raise Exception('f_4 is not trained')
        
        x_plot = np.linspace(250, 1700, 200)
        fig = plt.gcf()
        fig.set_size_inches((16, 8))
        scatter = plt.scatter(df[independent_variable], df['CCS'], c = df['Charge'], s = 0.01)
        power2, = plt.plot(x_plot, lr.f_2(x_plot) , 'b--', label = 'fit charge 2')
        power3, = plt.plot(x_plot, lr.f_3(x_plot) , 'b--', label = 'fit charge 3')
        power4, = plt.plot(x_plot, lr.f_4(x_plot) , 'b--', label = 'fit charge 4')
        plt.xlabel(independent_variable)
        plt.ylabel(r'CCA ($A^2$)')
        plt.title(f'Scatter plot: CCS vs {independent_variable}')
        legend1 = plt.legend(*scatter.legend_elements(), title = 'Charges')
        legend2 = plt.legend([power2, power3, power4], 
        [f'{np.round(popt[0], 2)}*x + {np.round(popt[1], 2)}' for popt in [lr.args_f2, lr.args_f3, lr.args_f4]], loc = 4)
        plt.gca().add_artist(legend1)

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
        i = 0
        for ax, df_it, f_i in zip(ax, [df[df['Charge']==2], df[df['Charge']==3], df[df['Charge']==4]],[lr.f_2, lr.f_3, lr.f_4]):
            sns.histplot((df_it['CCS']-f_i(df_it[independent_variable]))/f_i(df_it[independent_variable])*100, ax = ax)
            ax.set_xlabel('Residual %')
            ax.set_ylabel('Count')
            ax.set_title(f'Charge {i+2}')
            i += 1
    
    def test_set_plot(self, df, independent_variable = 'm/z'):
        if 'Charge' not in df:
            raise Exception('df doesnt contain charge')
        if 'm/z' not in df:
            raise Exception('df doesnt contain m/z')
        if self.f_2(0) is None:
            raise Exception('f_2 is not trained')
        if self.f_3(0) is None:
            raise Exception('f_3 is not trained')
        if self.f_4(0) is None:
            raise Exception('f_4 is not trained')

        self.predict(df, independent_variable=independent_variable)

        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 6))
        fig.suptitle('linear_ccs', fontsize = 12)
        res_rel = (df['CCS']-df['linear_ccs'])/df['linear_ccs']*100
        ax[0].hist(res_rel, bins = 50, label = f'MAD = {np.round(scipy.stats.median_abs_deviation(res_rel), 4)}')
        ax[0].set_xlabel('Relative Error of CCS')
        ax[0].set_ylabel('Counts')
        ax[0].set_title('Relative error of CCS w.r.t Ground Truth')
        ax[0].legend()

        corr, _ = scipy.stats.pearsonr(df['linear_ccs'],df['CCS'])
        ax[1].scatter(df['CCS'], df['linear_ccs'], label = f'Corr : {np.round(corr, 4)}', s = 0.1)
        ax[1].set_xlabel('CCS')
        ax[1].set_ylabel('Predicted CCS')
        ax[1].set_title('Scatter Plot CCS vs predicted CCS')
        ax[1].plot(np.arange(300,800), np.arange(300,800), 'b--')
        ax[1].legend()

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
        i = 0
        for ax, df_it in zip(ax, [df[df['Charge']==2], df[df['Charge']==3], df[df['Charge']==4]]):
            res_rel = (df_it['CCS']-df_it['linear_ccs'])/df_it['linear_ccs']*100
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
    lr = LinearRegression()
    lr.fit(df)
    lr.predict(df4)  
    lr.error_fit_plot(df4)
    lr.test_set_plot(df4)

# %%
