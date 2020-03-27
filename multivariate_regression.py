# -*- coding: utf-8 -*-
"""Multivariate regression objects for data analysis."""

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


class principal_component_regression():
    """Class for performing principal component regression."""

    def __init__(self, x, y):
        """
        Store input data and initialize results DataFrame.

        Parameters
        ----------
        x : ndarray
            Sample training data in the shape (n_samples, n_variables).
        y : ndarray
            Target values in the shape (n_samples,) for a single target or
            (n_samples, n_targets) for multiple targets.

        Returns
        -------
        None.

        """
        self.x = x
        self.y = y

        results_index = pd.MultiIndex.from_product(
            [['y', 'r2', 'mse'], ['c', 'cv']],
            names=['result_name', 'cal_or_crossval'])
        results_columns = pd.Index(
            [], name='n_components')
        self.pcr_results = pd.DataFrame([], index=results_index,
                                        columns=results_columns, dtype=object)
        self.pcr_objects = pd.DataFrame([], dtype='object',
                                        index=['pca', 'linear_model'])

    def pcr_fit(self, n_components, cv_percentage=10, **kwargs):
        """
        Perform principal component regression for one number of components.

        Currently, one PCA object is generated for every number of components.
        This is not necessary and might be changed in the future to save
        memory.

        Parameters
        ----------
        n_components : int
            Number of components kept after principal component analysis.
        cv_percentage : float, optional
            Percentage of data used for cross-validation. The default is 10.
        **kwargs :
            All **kwargs from sklearn.decomposition.PCA and
            sklearn.linear_model.LinearRegression are allowed, see the
            documentation of those classes for details.

        Returns
        -------
        DataFrame
            Contains the predicted sample values ('y') of the training data
            ('c'), the corresponding predictions ('y') after cross-validation
            ('cv'), the coefficient of determination ('r2') for 'c' and 'cv',
            and the mean squared error ('mse') for 'c' and 'cv' (given in the
            DataFrame index). The DataFrame columns give the number of
            components used to calculate the values.

        """
        # PCA options
        copy = kwargs.get('copy', True)
        whiten = kwargs.get('whiten', False)
        svd_solver = kwargs.get('svd_solver', 'auto')
        tol = kwargs.get('tol', 0.0)
        iterated_power = kwargs.get('iterated_power', 'auto')
        random_state = kwargs.get('random_state', None)

        # LinearRegression options
        fit_intercept = kwargs.get('fit_intercept', True)
        normalize = kwargs.get('normalize', False)
        copy_X = kwargs.get('copy_X', True)
        n_jobs = kwargs.get('n_jobs', None)

        # Create PCA objects
        self.pcr_objects.at['pca', n_components] = PCA(
            n_components=n_components, copy=copy, whiten=whiten,
            svd_solver=svd_solver, tol=tol, iterated_power=iterated_power,
            random_state=random_state)
        curr_pc_scores = self.pcr_objects.at[
            'pca', n_components].fit_transform(self.x)
        # self.pc_loadings = self.components_.T
        # * np.sqrt(pca.explained_variance_)
        # self.pc_explained_variance = self.explained_variance_ratio_

        # Create linear model objects
        self.pcr_objects.at['linear_model', n_components] = (
            linear_model.LinearRegression(fit_intercept=fit_intercept,
                                          normalize=normalize, copy_X=copy_X,
                                          n_jobs=n_jobs))
        # Build linear model
        self.pcr_objects.at['linear_model', n_components].fit(
            curr_pc_scores, self.y)
        # Predict sample values according to PCR model
        self.pcr_results[n_components] = pd.Series([], dtype=object)
        self.pcr_results.at[('y', 'c'), n_components] = self.pcr_objects.at[
            'linear_model', n_components].predict(curr_pc_scores)
        # Cross-validate the PCR model
        self.pcr_results.at[('y', 'cv'), n_components] = cross_val_predict(
            self.pcr_objects.at['linear_model', n_components], curr_pc_scores,
            self.y, cv=round(100/cv_percentage))
        # Calculate metrics for PCR model
        self.pcr_results.at[('r2', 'c'), n_components] = r2_score(
            self.y, self.pcr_results.at[('y', 'c'), n_components])
        self.pcr_results.at[('r2', 'cv'), n_components] = r2_score(
            self.y, self.pcr_results.at[('y', 'cv'), n_components])
        self.pcr_results.at[('mse', 'c'), n_components] = mean_squared_error(
            self.y, self.pcr_results.at[('y', 'c'), n_components])
        self.pcr_results.at[('mse', 'cv'), n_components] = mean_squared_error(
            self.y, self.pcr_results.at[('y', 'cv'), n_components])

        return self.pcr_results

    def pcr_sweep(self, max_components=20, **kwargs):
        """
        Perform PCR for all number of components between 1 and max_components.

        Parameters
        ----------
        max_components : int, optional
            The upper limit of components used for PCR. The default is 20.
        **kwargs :
            The same **kwargs as in self.pcr_fit.

        Returns
        -------
        DataFrame
            See Docstring of self.pcr_fit.

        """
        for ii in range(1, max_components+1):
            self.pcr_fit(ii, **kwargs)
        return self.pcr_results

    def predict(self, samples, n_components):
        """
        Predict unknown sample target values.

        Parameters
        ----------
        samples : ndarray
            Sample data in the shape (n_samples, n_variables).
        n_components : int
            Number of components used in the PCR model for the prediction.

        Returns
        -------
        prediction : ndarray
            Predicted target values in the shape (n_samples,) for a single
            target or (n_samples, n_targets) for multiple targets.

        """
        transformed_samples = self.pcr_objects.at[
            'pca', n_components].transform(samples)
        prediction = self.pcr_objects.at[
            'linear_model', n_components].predict(transformed_samples)
        return prediction

    def generate_plots(self, plot_names, **kwargs):
        """
        Generate some basic plots of principal component regression results.

        Parameters
        ----------
        plot_names : list of str
            List of plots to be generated. Allowed entries are
            'actual_vs_pred' (actual target values vs. predicted values),
            'r2_vs_comp' (coefficient of determination vs. number of
            components), 'mse_vs_comp' (mean squared error vs. number of
            components).
        **kwargs :
            n_components : int
                Needed for plot_name 'actual_vs_pred'.

        Returns
        -------
        plots : matplotlib plots
            Plot objects, I do not know if anything can be done with this.

        """
        plots = []
        if 'actual_vs_pred' in plot_names:
            n_components = kwargs.get('n_components')

            z_c = np.polyfit(
                self.y, self.pcr_results.at[('y', 'c'), n_components], 1)
            z_cv = np.polyfit(
                self.y, self.pcr_results.at[('y', 'cv'), n_components], 1)
            with plt.style.context(('ggplot')):
                fig1, ax1 = plt.subplots(figsize=(9, 5))
                ax1.scatter(self.y, self.pcr_results.at[
                    ('y', 'c'), n_components], c='red', edgecolors='k')
                ax1.scatter(self.y, self.pcr_results.at[
                    ('y', 'cv'), n_components], c='blue', edgecolors='k')
                ax1.plot(self.y, z_c[1]+z_c[0]*self.y, c='red', linewidth=1)
                ax1.plot(self.y, z_cv[1]+z_cv[0]*self.y, c='blue', linewidth=1)
                ax1.plot(self.y, self.y, color='green', linewidth=1)
                plt.title('$R^{2}$ (CV): '+str(
                    self.pcr_results.loc[('r2', 'cv'), n_components]))
                plt.xlabel('Measured')
                plt.ylabel('Predicted')
            plots.append(fig1)
        if 'r2_vs_comp' in plot_names:
            with plt.style.context(('ggplot')):
                fig2, ax2 = plt.subplots(figsize=(9, 5))
                ax2.plot(
                    self.pcr_results.loc[
                        'r2', 'c'], linestyle='--', marker='o')
                ax2.plot(
                    self.pcr_results.loc[
                        'r2', 'cv'], linestyle='--', marker='o')
                plt.ylabel('$R^{2}$')
                plt.xlabel('Number of components')
            plots.append(fig2)
        if 'mse_vs_comp' in plot_names:
            with plt.style.context(('ggplot')):
                fig3, ax3 = plt.subplots(figsize=(9, 5))
                ax3.plot(
                    self.pcr_results.loc[
                        'mse', 'c'], linestyle='--', marker='o')
                ax3.plot(
                    self.pcr_results.loc[
                        'mse', 'cv'], linestyle='--', marker='o')
                plt.ylabel('MSE')
                plt.xlabel('Number of components')
            plots.append(fig3)

        return plots


class pls_regression():
    """Class for performing principal component regression."""

    def __init__(self, x, y):
        """
        Store input data and initialize results DataFrame.

        Parameters
        ----------
        x : ndarray
            Sample training data in the shape (n_samples, n_variables).
        y : ndarray
            Target values in the shape (n_samples,) for a single target or
            (n_samples, n_targets) for multiple targets.

        Returns
        -------
        None.

        """
        self.x = x
        self.y = y

        results_index = pd.MultiIndex.from_product(
            [['y', 'r2', 'mse'], ['c', 'cv']],
            names=['result_name', 'cal_or_crossval'])
        results_columns = pd.Index(
            [], name='n_components')
        self.plsr_results = pd.DataFrame([], index=results_index,
                                         columns=results_columns, dtype=object)
        self.plsr_objects = pd.Series([], dtype='object')

    def plsr_fit(self, n_components, cv_percentage=10, **kwargs):
        """
        Perform partial least squares regression for one number of components.

        Currently, one PLSR object is generated for every number of components.
        I do not know if this is necessary, might be checked in the future.

        Parameters
        ----------
        n_components : int
            Number of components kept after PLSR.
        cv_percentage : float, optional
            Percentage of data used for cross-validation. The default is 10.
        **kwargs :
            All **kwargs from sklearn.cross_decomposition.PLSRegression are
            allowed, see the documentation of those classes for details.

        Returns
        -------
        DataFrame
            Contains the predicted sample values ('y') of the training data
            ('c'), the corresponding predictions ('y') after cross-validation
            ('cv'), the coefficient of determination ('r2') for 'c' and 'cv',
            and the mean squared error ('mse') for 'c' and 'cv' (given in the
            DataFrame index). The DataFrame columns give the number of
            components used to calculate the values.

        """
        # Create PLSR models
        self.plsr_objects.at[n_components] = PLSRegression(
            n_components=n_components, **kwargs)
        self.plsr_objects.at[n_components].fit_transform(self.x, y=self.y)
        # Predict sample values according to PLSR model
        self.plsr_results[n_components] = pd.Series([], dtype=object)
        self.plsr_results.at[('y', 'c'), n_components] = self.predict(
            self.x, n_components)
        # Cross-validate the PCR model
        self.plsr_results.at[('y', 'cv'), n_components] = cross_val_predict(
            self.plsr_objects.at[n_components], self.x, self.y,
            cv=round(100/cv_percentage))

        # Calculate metrics for PLSR model
        self.plsr_results.at[('r2', 'c'), n_components] = r2_score(
            self.y, self.plsr_results.at[('y', 'c'), n_components])
        self.plsr_results.at[('r2', 'cv'), n_components] = r2_score(
            self.y, self.plsr_results.at[('y', 'cv'), n_components])
        self.plsr_results.at[('mse', 'c'), n_components] = mean_squared_error(
            self.y, self.plsr_results.at[('y', 'c'), n_components])
        self.plsr_results.at[('mse', 'cv'), n_components] = mean_squared_error(
            self.y, self.plsr_results.at[('y', 'cv'), n_components])

        return self.plsr_results

    def plsr_sweep(self, max_components=20, **kwargs):
        """
        Perform PLSR for all number of components between 1 and max_components.

        Parameters
        ----------
        max_components : int, optional
            The upper limit of components used for PLSR. The default is 20.
        **kwargs :
            The same **kwargs as in self.plsr_fit.

        Returns
        -------
        DataFrame
            See Docstring of self.plsr_fit.

        """
        for ii in range(1, max_components+1):
            self.plsr_fit(ii, **kwargs)
        return self.plsr_results

    def predict(self, samples, n_components):
        """
        Predict unknown sample target values.

        Parameters
        ----------
        samples : ndarray
            Sample data in the shape (n_samples, n_variables).
        n_components : int
            Number of components used in the PLSR model for the prediction.

        Returns
        -------
        prediction : ndarray
            Predicted target values in the shape (n_samples,) for a single
            target or (n_samples, n_targets) for multiple targets.

        """
        return self.plsr_objects.at[n_components].predict(samples)

    def generate_plots(self, plot_names, **kwargs):
        """
        Generate some basic plots of partial least squares regression results.

        Parameters
        ----------
        plot_names : list of str
            List of plots to be generated. Allowed entries are
            'actual_vs_pred' (actual target values vs. predicted values),
            'r2_vs_comp' (coefficient of determination vs. number of
            components), 'mse_vs_comp' (mean squared error vs. number of
            components).
        **kwargs :
            n_components : int
                Needed for plot_name 'actual_vs_pred'.

        Returns
        -------
        plots : matplotlib plots
            Plot objects, I do not know if anything can be done with this.

        """
        plots = []
        if 'actual_vs_pred' in plot_names:
            n_components = kwargs.get('n_components')

            z_c = np.polyfit(
                self.y, self.plsr_results.at[('y', 'c'), n_components], 1)
            z_cv = np.polyfit(
                self.y, self.plsr_results.at[('y', 'cv'), n_components], 1)
            with plt.style.context(('ggplot')):
                fig1, ax1 = plt.subplots(figsize=(9, 5))
                ax1.scatter(self.y, self.plsr_results.at[
                    ('y', 'c'), n_components], c='red', edgecolors='k')
                ax1.scatter(self.y, self.plsr_results.at[
                    ('y', 'cv'), n_components], c='blue', edgecolors='k')
                ax1.plot(self.y, z_c[1]+z_c[0]*self.y, c='red', linewidth=1)
                ax1.plot(self.y, z_cv[1]+z_cv[0]*self.y, c='blue', linewidth=1)
                ax1.plot(self.y, self.y, color='green', linewidth=1)
                plt.title('$R^{2}$ (CV): '+str(
                    self.plsr_results.loc[('r2', 'cv'), n_components]))
                plt.xlabel('Measured')
                plt.ylabel('Predicted')
            plots.append(fig1)
        if 'r2_vs_comp' in plot_names:
            with plt.style.context(('ggplot')):
                fig2, ax2 = plt.subplots(figsize=(9, 5))
                ax2.plot(
                    self.plsr_results.loc[
                        'r2', 'c'], linestyle='--', marker='o')
                ax2.plot(
                    self.plsr_results.loc[
                        'r2', 'cv'], linestyle='--', marker='o')
                plt.ylabel('$R^{2}$')
                plt.xlabel('Number of components')
            plots.append(fig2)
        if 'mse_vs_comp' in plot_names:
            with plt.style.context(('ggplot')):
                fig3, ax3 = plt.subplots(figsize=(9, 5))
                ax3.plot(
                    self.plsr_results.loc[
                        'mse', 'c'], linestyle='--', marker='o')
                ax3.plot(
                    self.plsr_results.loc[
                        'mse', 'cv'], linestyle='--', marker='o')
                plt.ylabel('MSE')
                plt.xlabel('Number of components')
            plots.append(fig3)

        return plots


if __name__ == "__main__":

    def preprocess_pcr(x):
        # Preprocessing (1): first derivative
        d1X = savgol_filter(x, 25, polyorder=5, deriv=1)
        # Preprocess (2) Standardize features by removing the mean and scaling
        # to unit variance
        Xstd = StandardScaler().fit_transform(d1X[:, :])
        return Xstd

    def preprocess_pls(x):
        X2 = savgol_filter(x, 17, polyorder=2, deriv=2)
        return X2

    # import data
    data = pd.read_csv('../../yy_Test-Skripte_alte Skripte/PCR_data/'+
                       'peach_spectra+brixvalues.csv')
    X = data.values[:, 1:]
    Y = data['Brix']
    wl = np.arange(1100, 2300, 2) # wavelengths

    #########
    # PCR
    #########

    # Preprocess data and build model
    X_PCR = preprocess_pcr(X)
    pcr = principal_component_regression(X_PCR, Y)
    pcr_results = pcr.pcr_sweep(max_components=20)

    # Plot analyzed data
    with plt.style.context(('ggplot')):
        plt.plot(wl, X_PCR.T)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absorbance')
    plt.show()

    # plot PC regression results
    pcr_components = 6
    pcr.generate_plots(['actual_vs_pred', 'r2_vs_comp', 'mse_vs_comp'],
                       n_components=pcr_components)

    #########
    # PLSR
    #########

    # Preprocess data and build model
    X_PLSR = preprocess_pls(X)
    plsr = pls_regression(X_PLSR, Y)
    plsr_results = plsr.plsr_sweep(max_components=40)

    # Plot analyzed data
    plt.figure(figsize=(8, 4.5))
    with plt.style.context(('ggplot')):
        plt.plot(wl, X_PLSR.T)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('D2 Absorbance')
        plt.show()

    # plot PLS regression results
    plsr_components = 7
    plsr.generate_plots(['actual_vs_pred', 'r2_vs_comp', 'mse_vs_comp'],
                        n_components=plsr_components)
    plt.show()
