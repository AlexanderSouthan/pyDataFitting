# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 14:50:48 2023

@author: southan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


class pls_regression():
    """Class for performing principal component regression."""

    def __init__(self, x, y, scale_std=False):
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
        self.x_raw = x
        self.y = y
        self.scale_std = scale_std

        self.scaler = StandardScaler(with_std=self.scale_std)
        self.x = self.scaler.fit_transform(self.x_raw)

        results_index = pd.MultiIndex.from_product(
            [['plsr_objects', 'y', 'r2', 'mse', 'Hotelling_T2'], ['c', 'cv']],
            names=['result_name', 'cal_or_crossval'])
        results_columns = pd.Index(
            [], name='n_components')
        self.plsr_results = pd.DataFrame([], index=results_index,
                                         columns=results_columns, dtype='object')

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
        self.plsr_results.at[('plsr_objects', 'c'), n_components] = PLSRegression(
            n_components=n_components, **kwargs)
        self.plsr_results.at[('plsr_objects', 'c'), n_components].fit_transform(self.x, y=self.y)
        # Predict sample values according to PLSR model
        self.plsr_results.at[('y', 'c'), n_components] = self.predict(
            self.x, n_components, scale=False)
        # Cross-validate the PCR model
        self.plsr_results.at[('y', 'cv'), n_components] = np.squeeze(
            cross_val_predict(
                self.plsr_results.at[('plsr_objects', 'c'), n_components], self.x, self.y,
                cv=round(100/cv_percentage)))

        # Calculate metrics for PLSR model
        self.plsr_results.at[('r2', 'c'), n_components] = r2_score(
            self.y, self.plsr_results.at[('y', 'c'), n_components],
            multioutput='raw_values')
        self.plsr_results.at[('r2', 'cv'), n_components] = r2_score(
            self.y, self.plsr_results.at[('y', 'cv'), n_components],
            multioutput='raw_values')
        self.plsr_results.at[('mse', 'c'), n_components] = mean_squared_error(
            self.y, self.plsr_results.at[('y', 'c'), n_components],
            multioutput='raw_values')
        self.plsr_results.at[('mse', 'cv'), n_components] = mean_squared_error(
            self.y, self.plsr_results.at[('y', 'cv'), n_components],
            multioutput='raw_values')

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

    def predict(self, samples, n_components, scale=True):
        """
        Predict unknown sample target values.

        Parameters
        ----------
        samples : ndarray
            Sample data in the shape (n_samples, n_variables).
        n_components : int
            Number of components used in the PLSR model for the prediction.
        scale : bool, optional
            Defines if the sample data is scaled like the input data or not.
            If called from within the class, should be False. Default is True.

        Returns
        -------
        prediction : ndarray
            Predicted target values in the shape (n_samples,) for a single
            target or (n_samples, n_targets) for multiple targets.

        """
        if scale:
            samples = self.scaler.transform(samples)

        return np.squeeze(
            self.plsr_results.at[('plsr_objects', 'c'), n_components].predict(samples))

    def generate_plots(self, plot_names, response_number=0, **kwargs):
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
        response_number : int, optional
            Defines the index of the response from self.y to be plotted. 
            Default is 0.
        **kwargs :
            n_components : int
                Needed for plot_name 'actual_vs_pred'.
            cv : boolean
                State if cross-validation data should be plotted, too.
                Default is False.
        Returns
        -------
        plots : matplotlib plots
            Plot objects, I do not know if anything can be done with this.

        """
        plots = []
        plot_cv_data = kwargs.get('cv', False)
        if 'actual_vs_pred' in plot_names:
            n_components = kwargs.get('n_components')

            if len(self.y.shape) == 1:
                curr_y = self.y
            else:
                curr_y = self.y[:, response_number]
                
            if len(self.plsr_results.at[('y', 'c'), n_components].shape) == 1:
                curr_c = self.plsr_results.at[('y', 'c'), n_components]
            else:
                curr_c = self.plsr_results.at[('y', 'c'), n_components][
                    :, response_number]
                
            if len(self.plsr_results.at[('y', 'cv'), n_components].shape) == 1:
                curr_cv = self.plsr_results.at[('y', 'cv'), n_components]
            else:
                curr_cv = self.plsr_results.at[('y', 'cv'), n_components][
                    :, response_number]

            z_c = np.polyfit(
                curr_y, curr_c, 1)
            z_cv = np.polyfit(
                curr_y, curr_cv, 1)
            with plt.style.context(('ggplot')):
                fig1, ax1 = plt.subplots(figsize=(9, 5))
                ax1.scatter(curr_y, curr_c, c='red', edgecolors='k')
                if plot_cv_data:
                    ax1.scatter(curr_y, curr_cv, c='blue', edgecolors='k')
                    ax1.plot(curr_y, z_cv[1]+z_cv[0]*curr_y, c='blue',
                             linewidth=1)
                ax1.plot(curr_y, z_c[1]+z_c[0]*curr_y, c='red', linewidth=1)
                ax1.plot(curr_y, curr_y, color='green', linewidth=1)
                plt.title('$R^{2}$ (CV): '+str(
                    self.plsr_results.loc[('r2', 'cv'), n_components]))
                plt.xlabel('Measured')
                plt.ylabel('Predicted')
            plots.append(fig1)
        if 'r2_vs_comp' in plot_names:
            if len(self.plsr_results.loc['r2', 'c'][1]) == 1:
                curr_c_r2 = self.plsr_results.loc['r2', 'c']
            else:
                curr_c_r2 = pd.DataFrame(
                    self.plsr_results.loc['r2', 'c'].tolist()
                    ).iloc[:, response_number]

            if len(self.plsr_results.loc['r2', 'cv'][1]) == 1:
                curr_cv_r2 = self.plsr_results.loc['r2', 'cv']
            else:
                curr_cv_r2 = pd.DataFrame(
                    self.plsr_results.loc['r2', 'cv'].tolist()
                    ).iloc[:, response_number]

            with plt.style.context(('ggplot')):
                fig2, ax2 = plt.subplots(figsize=(9, 5))
                ax2.plot(
                    curr_c_r2, linestyle='--', marker='o')
                if plot_cv_data:
                    ax2.plot(
                        curr_cv_r2, linestyle='--', marker='o')
                plt.ylabel('$R^{2}$')
                plt.xlabel('Number of components')
            plots.append(fig2)
        if 'mse_vs_comp' in plot_names:
            if len(self.plsr_results.loc['mse', 'c'][1]) == 1:
                curr_c_mse = self.plsr_results.loc['mse', 'c']
            else:
                curr_c_mse = pd.DataFrame(
                    self.plsr_results.loc['mse', 'c'].tolist()
                    ).iloc[:, response_number]

            if len(self.plsr_results.loc['mse', 'cv'][1]) == 1:
                curr_cv_mse = self.plsr_results.loc['mse', 'cv']
            else:
                curr_cv_mse = pd.DataFrame(
                    self.plsr_results.loc['mse', 'cv'].tolist()
                    ).iloc[:, response_number]
            
            with plt.style.context(('ggplot')):
                fig3, ax3 = plt.subplots(figsize=(9, 5))
                ax3.plot(
                    curr_c_mse, linestyle='--', marker='o')
                if plot_cv_data:
                    ax3.plot(
                        curr_cv_mse, linestyle='--', marker='o')
                plt.ylabel('MSE')
                plt.xlabel('Number of components')
            plots.append(fig3)

        return plots
