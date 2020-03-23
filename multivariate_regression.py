# -*- coding: utf-8 -*-
"""
Multivariate regression objects for data analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

from sys import stdout

class principal_component_regression(PCA):

    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.recent_pcr_results = pd.DataFrame(
            [], index=np.arange(self.x.shape[0]), columns=['y_c', 'y_cv'],
            dtype='float64')
        self.recent_pcr_metrics = pd.Series([], dtype='float64')

    def PCR_fit(self, n_components, cv_percentage=10, **kwargs):
        # PCA options
        copy = kwargs.get('copy', True)
        whiten = kwargs.get('whiten' ,False)
        svd_solver = kwargs.get('svd_solver', 'auto')
        tol = kwargs.get('tol', 0.0)
        iterated_power = kwargs.get('iterated_power', 'auto')
        random_state = kwargs.get('random_state', None)

        # LinearRegression options
        fit_intercept = kwargs.get('fit_intercept', True)
        normalize = kwargs.get('normalize', False)
        copy_X = kwargs.get('copy_X', True)
        n_jobs = kwargs.get('n_jobs', None)

        self.pca(n_components, copy=copy, whiten=whiten,
                 svd_solver=svd_solver, tol=tol, iterated_power=iterated_power,
                 random_state=random_state)

        # Create linear regression object
        self.PCR_model = linear_model.LinearRegression(
            fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X,
            n_jobs=n_jobs)
        # Fit
        self.PCR_model.fit(self.pc_scores, self.y)
        # Calibration
        self.recent_pcr_results['y_c'] = self.PCR_model.predict(self.pc_scores)
        # Cross-validation
        self.recent_pcr_results['y_cv'] = cross_val_predict(
            self.PCR_model, self.pc_scores, self.y,
            cv=round(100/cv_percentage))
        # Calculate scores for calibration and cross-validation
        self.recent_pcr_metrics.loc['r2_c'] = r2_score(
            self.y, self.recent_pcr_results['y_c'])
        self.recent_pcr_metrics.loc['r2_cv'] = r2_score(
            self.y, self.recent_pcr_results['y_cv'])
        # Calculate mean square error for calibration and cross validation
        self.recent_pcr_metrics.loc['mse_c'] = mean_squared_error(
            self.y, self.recent_pcr_results['y_c'])
        self.recent_pcr_metrics.loc['mse_cv'] = mean_squared_error(
            self.y, self.recent_pcr_results['y_cv'])

        return (self.recent_pcr_results, self.recent_pcr_metrics)

    def PCR_sweep(self, max_components=20, **kwargs):
        results_index = pd.MultiIndex.from_product(
            [['c', 'cv'], np.arange(1, max_components+1)],
            names=['type', 'n_components'])
        self.pcr_sweep_results = pd.DataFrame(
            [], index=results_index,
            columns=np.arange(self.x.shape[0]), dtype='float64')
        self.pcr_sweep_metrics = pd.DataFrame(
            [], index=np.arange(1, max_components+1),
            columns=['r2_c', 'r2_cv', 'mse_c', 'mse_cv'], dtype='float64')

        for ii in range(1, 21):
            curr_results = self.PCR_fit(ii, **kwargs)
            self.pcr_sweep_results.loc[('c', ii)] = curr_results[0].loc[
                :,'y_c']
            self.pcr_sweep_results.loc[('cv', ii)] = curr_results[0].loc[
                :,'y_cv']
            self.pcr_sweep_metrics.loc[ii, :] = curr_results[1]
            
        return (self.pcr_sweep_results, self.pcr_sweep_metrics)

    def pca(self, n_components, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.pc_scores = self.fit_transform(self.x)
        self.pc_loadings = self.components_.T  # * np.sqrt(pca.explained_variance_)
        self.pc_explained_variance = self.explained_variance_ratio_

    def predict(self, samples):
        transformed_samples = self.transform(samples)
        prediction = self.PCR_model.predict(transformed_samples)
        
        return prediction

def optimise_pls_cv(X, y, n_comp, plot_components=True):

    '''Run PLS including a variable number of components, up to n_comp,
       and calculate MSE '''

    mse = []
    component = np.arange(1, n_comp)

    for i in component:
        pls = PLSRegression(n_components=i)

        # Cross-validation
        y_cv = cross_val_predict(pls, X, y, cv=10)

        mse.append(mean_squared_error(y, y_cv))

        comp = 100*(i+1)/40
        # Trick to update status on the same line
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")

    # Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    print("Suggested number of components: ", msemin+1)
    stdout.write("\n")

    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(mse), '-v', color = 'blue', mfc='blue')
            plt.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('MSE')
            plt.title('PLS')
            plt.xlim(left=-1)

        plt.show()

    # Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components=msemin+1)

    # Fir to the entire dataset
    pls_opt.fit(X, y)
    y_c = pls_opt.predict(X)

    # Cross-validation
    y_cv = cross_val_predict(pls_opt, X, y, cv=10)

    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)

    # Calculate mean squared error for calibration and cross validation
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)

    print('R2 calib: %5.3f'  % score_c)
    print('R2 CV: %5.3f'  % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)

    # Plot regression and figures of merit
    rangey = max(y) - min(y)
    rangex = max(y_c) - min(y_c)

    # Fit a line to the CV vs response
    z = np.polyfit(y, y_c, 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_c, y, c='red', edgecolors='k')
        #Plot the best fit line
        ax.plot(np.polyval(z,y), y, c='blue', linewidth=1)
        #Plot the ideal 1:1 line
        ax.plot(y, y, color='green', linewidth=1)
        plt.title('$R^{2}$ (CV): '+str(score_cv))
        plt.xlabel('Predicted $^{\circ}$Brix')
        plt.ylabel('Measured $^{\circ}$Brix')

        plt.show()

    return

def preprocess_pcr(x):
    # Preprocessing (1): first derivative
    d1X = savgol_filter(x, 25, polyorder=5, deriv=1)
    # Preprocess (2) Standardize features by removing the mean and scaling to unit variance
    Xstd = StandardScaler().fit_transform(d1X[:, :])
    return Xstd

def preprocess_pls(x):
    X2 = savgol_filter(x, 17, polyorder = 2,deriv=2)
    return X2

if __name__ == "__main__":
    data = pd.read_csv('../../yy_Test-Skripte_alte Skripte/PCR_data/peach_spectra+brixvalues.csv')
    X = data.values[:,1:]
    y = data['Brix']
    wl = np.arange(1100,2300,2) # wavelengths
    # Plot absorbance spectra
    with plt.style.context(('ggplot')):
        plt.plot(wl, X.T)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absorbance')
    plt.show()

    X_processed = preprocess_pcr(X)

    pcr = principal_component_regression(X_processed, y)
    pcr_results, pcr_metrics = pcr.PCR_sweep()

    with plt.style.context(('ggplot')):
        plt.figure()
        plt.plot(pcr_metrics['r2_c'], linestyle='--', marker='o')
        plt.plot(pcr_metrics['r2_cv'], linestyle='--', marker='o')

        plt.figure()
        plt.plot(pcr_metrics['mse_c'], linestyle='--', marker='o')
        plt.plot(pcr_metrics['mse_cv'], linestyle='--', marker='o')

    # Regression plot
    z_c = np.polyfit(y, pcr_results.loc[('c', 6)], 1)
    z_cv = np.polyfit(y, pcr_results.loc[('cv', 6)], 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y, pcr_results.loc[('c', 6)], c='red', edgecolors='k')
        ax.scatter(y, pcr_results.loc[('cv', 6)], c='blue', edgecolors='k')
        ax.plot(y, z_c[1]+z_c[0]*y, c='red', linewidth=1)
        ax.plot(y, z_cv[1]+z_cv[0]*y, c='blue', linewidth=1)
        ax.plot(y, y, color='green', linewidth=1)
        plt.title('$R^{2}$ (CV): '+str(pcr_metrics.loc[6, 'r2_cv']))
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
    plt.show()

#########
# PLS
#########

# Plot second derivative
X2 = preprocess_pls(X)

plt.figure(figsize=(8,4.5))
with plt.style.context(('ggplot')):
    plt.plot(wl, X2.T)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('D2 Absorbance')
    plt.show()

optimise_pls_cv(X2,y, 40, plot_components=True)