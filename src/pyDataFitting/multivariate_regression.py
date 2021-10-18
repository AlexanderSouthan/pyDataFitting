# -*- coding: utf-8 -*-
"""Multivariate regression objects for data analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.cm import rainbow
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

from little_helpers.statsmodel_wrapper import statsmodel_wrapper


class principal_component_regression():
    """Class for performing principal component regression."""

    def __init__(self, x, y=None, x_names=None, y_names=None,
                 sample_names=None, scale_std=False):
        """
        Store input data and initialize results DataFrame.

        Parameters
        ----------
        x : ndarray or pandas DataFrame
            Sample training data in the shape (n_samples, n_variables). Data is
            mean centered automatically, so it is not necessary to do that
            before. If a pandas DataFrame, the index is used for sample naming
            and the columns for variable naming, so potentially given values by
            x_names and sample_names are ignored.
        y : ndarray, pandas DataFrame or None, optional
            Target values in the shape (n_samples,) for a single target or
            (n_samples, n_targets) for multiple targets. Must be given if a
            regression is to be performed, otherwise only a PCA on x makes
            sense. Default is None in which case the regression results are
            empty and meaningless. If a pandas DataFrame, the columns are used
            for target naming, so potentially given values by y_names are
            ignored.
        x_names : list of str or None, optional
            A list containing the names of the n_variables factors. Default is
            None which results in numbered variables.
        y_names : list of str or None, optional
            A list containing the names of the n_targets responses. Default is
            None which results in response names of 'response_1', 'response_2'
            etc.
        sample_names : list of str or None, optional
            A list containing the n_samples sample names. Default is None which
            results in sample names of 'sample_1', 'sample_2' etc.
        scale_std : bool, optional
            True means the data will be scaled to unit variance. Default is
            False.

        Returns
        -------
        None.

        """
        self.scale_std = scale_std
        self.n_samples, self.n_variables = x.shape
        if y is None:
            self.n_responses = 0
        elif (y is not None) and (len(np.squeeze(y).shape) == 1):
            self.n_responses = 1
        else:
            self.n_responses = y.shape[1]

        if isinstance(y, np.ndarray) or (y is None):
            if (y_names is not None) and (len(y_names) == self.n_responses):
                self.y_names = y_names
            elif y_names is None:
                self.y_names = ['response_{}'.format(ii) for ii in np.arange(
                    1, self.n_responses+1)]
            else:
                raise ValueError(
                    'Number of response names does not match number of '
                    'responses. Number of responses is {} and response name '
                    'number is {}.'.format(self.n_responses, len(x_names)))

            self.response_index = pd.Index(self.y_names, name='Response name')
        elif isinstance(y, pd.DataFrame):
            self.y_names = y.columns.to_numpy()
            self.response_index = y.columns
        else:
            raise TypeError('y must either be np.ndarray or '
                            'pd.DataFrame or None.')

        if isinstance(x, np.ndarray):
            if (x_names is not None) and (len(x_names) == self.n_variables):
                self.x_names = x_names
            elif x_names is None:
                self.x_names = ['factor_{}'.format(ii) for ii in np.arange(
                    1, self.n_variables+1)]
            else:
                raise ValueError(
                    'Number of factor names does not match number of factors. '
                    'Number of factors is {} and factor name number is '
                    '{}.'.format(self.n_variables, len(x_names)))

            if (sample_names is not None) and (
                    len(sample_names) == self.n_samples):
                self.sample_names = sample_names
            elif sample_names is None:
                self.sample_names = [
                    'sample_{}'.format(ii) for ii in np.arange(
                        1, self.n_samples+1)]
            else:
                raise ValueError(
                    'Number of sample names does not match number of samples. '
                    'Number of samples is {} and sample name number is {}'
                    '.'.format(self.n_samples, len(sample_names)))

            # Construct indices for later use in the DataFrames that collect
            # the different PCA and PCR results
            self.sample_index = pd.Index(self.sample_names, name='Sample name')
            self.var_index = pd.Index(self.x_names, name='Variable name')
        elif isinstance(x, pd.DataFrame):
            self.sample_index = x.index
            self.var_index = x.columns
        else:
            raise TypeError('x must either be np.ndarray or '
                            'pd.DataFrame.')

        y_c_index = pd.MultiIndex.from_product(
            [self.response_index, self.sample_index.values],
            names=['response name', 'sample number'])
        metrics_index = pd.MultiIndex.from_product(
            [self.response_index, ['r2_c', 'r2_cv', 'rmse_c', 'rmse_cv']],
            names=['response name', 'metrics type'])

        self.scaler = StandardScaler(with_std=self.scale_std)
        self.x = pd.DataFrame(self.scaler.fit_transform(x),
                              index=self.sample_index, columns=self.var_index)

        if (y is not None) and (len(y) == len(x)):
            self.y = pd.DataFrame(
                y, index=self.sample_index, columns=self.response_index)
        elif y is None:
            self.y = None
        else:
            raise ValueError(
                'Number of responses does not match number of samples. '
                'Number of responses is {} and sample number is {}.'.format(
                    self.n_responses, self.n_samples))

        # List that will collect PCA component numbers that already
        # were calculated
        self.computed_components = [0]

        self.pca_objects = pd.Series([], dtype='object')
        self.pca_scores = pd.DataFrame([])
        self.pca_eigenvalues = pd.Series([], dtype='float64')
        self.pca_eigenvectors = pd.DataFrame([])
        self.pca_loadings = pd.DataFrame([])
        self.pca_explained_variance = pd.DataFrame([], columns=['each', 'cum'])
        self.pca_results = pd.DataFrame([],
                                        index=['Hotelling_T2', 'Q_residuals'])
        self.pcr_models = pd.DataFrame([], dtype='object',
                                       columns=self.response_index)
        self.pcr_params = pd.DataFrame([], dtype='object',
                                       columns=self.response_index)
        self.pcr_used_pcs = pd.DataFrame([], dtype='object',
                                         columns=self.response_index)
        self.pcr_corr_coef = pd.DataFrame([], columns=self.response_index)
        self.pcr_y_c = pd.DataFrame([], index=y_c_index)
        self.pcr_y_cv = pd.DataFrame([], index=y_c_index)
        self.pcr_metrics = pd.DataFrame([], index=metrics_index)

    def perform_pca(self, n_components, **kwargs):
        """
        Actually perform the principal component analysis.

        Parameters
        ----------
        n_components : int
            Number of components kept after principal component analysis.
        **kwargs :
            All **kwargs from sklearn.decomposition.PCA are allowed, see the
            documentation of that class for details.

        Returns
        -------
        None.

        """
        # Check if PCA with n_components was already done before. If so, no
        # calculation is performed because the data is already there.
        if n_components not in self.computed_components:
            self.pca_objects[n_components] = PCA(
                n_components=n_components, **kwargs)
            curr_scores = self.pca_objects[
                    n_components].fit_transform(self.x)

            # Check if any new information is generated by the calculations.
            # If that is the case, scores, loadings, etc. are stored in the
            # corresponding attributes.
            if n_components > max(self.computed_components):
                pc_index = pd.Index(np.arange(1, n_components+1),
                                    name='PC number')

                self.pca_scores = pd.DataFrame(curr_scores,
                                               index=self.sample_index,
                                               columns=pc_index)
                self.pca_eigenvalues = pd.Series(
                    self.pca_objects[n_components].explained_variance_,
                    index=pc_index)
                self.pca_eigenvectors = pd.DataFrame(
                    self.pca_objects[n_components].components_.T,
                    index=self.var_index,
                    columns=pc_index)
                self.pca_loadings = self.pca_eigenvectors * np.sqrt(
                    self.pca_eigenvalues)
                self.pca_explained_variance = pd.DataFrame(
                    self.pca_objects[n_components].explained_variance_ratio_,
                    columns=['each'], index=pc_index)
                self.pca_explained_variance['cum'] = (
                    self.pca_explained_variance['each'].cumsum())

            self.pca_results[n_components] = pd.Series([], dtype='object')
            # self.pc_loadings = self.components_.T
            # * np.sqrt(pca.explained_variance_)

            # Some PCA metrics are calculated. The results should be checked,
            # calculation is still experimental. Also the attribute pca_results
            # should be improved in the future.
            # self.pca_results.at['Hotelling_T2', n_components] = np.array(
            #     [sample.dot(self.pca_eigenvectors.loc[:, :n_components]).dot(
            #         np.diag(self.pca_eigenvalues ** -1)).dot(
            #             self.pca_eigenvectors.loc[:, :n_components].T).dot(
            #                 sample.T) for sample in self.x])
            # self.pca_results.at['Q_residuals', n_components] = np.array(
            #     [sample.dot(np.identity(self.x.shape[1])-self.pca_eigenvectors.loc[
            #         :, :n_components].dot(self.pca_eigenvectors.loc[
            #             :, :n_components].T)).dot(
            #                 sample.T) for sample in self.x])

            self.computed_components.append(n_components)

    def reconstruct_input(self):
        """
        Reconstruct the original input data.

        The input data are reconstructed from the scaled data using the
        standard deviation (if self.scale.std == true) and the
        mean data from self.scaler. The input data is not needed within this
        class, so the input data is not stored in order to save memory.

        Returns
        -------
        reconstructed_input : pandas DataFrame
            A DataFrame in the same format as self.x containing the input data.

        """
        reconstructed_input = pd.DataFrame(
            self.scaler.inverse_transform(self.x.copy()),
            index=self.x.index, columns=self.x.columns)
        return reconstructed_input

    def reconstruct_data(self, used_pcs):
        """
        Reconstruct the data from scores and eigenvectors.

        The result is the original data without the contents of the PCA error
        matrix, so can also be understood as a PCA based smoothing.

        Parameters
        ----------
        used_pcs : int or list of int
            If an int is given, the according number of principal components
            is used for data reconstruction based on the order of explained
            variance. If a list of int is given, the pricipal components given
            in the list will be used for data reconstruction, based on the
            explained variance ordering and the first PC having number 1.

        Returns
        -------
        reconstructed_data : pandas DataFrame
            The recomstructed data in the same format as self.x.

        """
        if isinstance(used_pcs, int):
            max_component = used_pcs
            pc_list = np.arange(1, used_pcs+1)
        elif isinstance(used_pcs, list) and all(
                [isinstance(ii, int) for ii in used_pcs]):
            max_component = max(used_pcs)
            pc_list = used_pcs
        else:
            raise TypeError('No valid value for used_pcs given. Allowed '
                            'inputs are an integer or a list of integers.')

        if max_component > max(self.computed_components):
            self.perform_pca(max_component)

        reconstructed_data = np.dot(
            self.pca_scores.loc[:, pc_list],
            self.pca_eigenvectors.loc[:, pc_list].T)

        reconstructed_data = pd.DataFrame(
            self.scaler.inverse_transform(reconstructed_data),
            index=self.x.index, columns=self.x.columns)

        return reconstructed_data

    def pcr_fit(self, cv_percentage=20, mode='exp_var',
                **kwargs):
        """
        Perform principal component regression for one number of components.

        One PCA object is generated for every number of components.
        This is necessary in order to be able to use the PCA transform method
        in self.predict. For example, the predicted sample values ('y') of the
        training data ('c'), the corresponding predictions ('y') after
        cross-validation ('cv'), the coefficient of determination ('r2') for
        'c' and 'cv', and the root mean squared error ('rmse') for 'c' and 'cv'
        are calculated.

        Parameters
        ----------
        cv_percentage : float, optional
            Percentage of data used for cross-validation. The default is 20.
        mode : str, optional
            Determines if PCs used for regression are selected based on
            explained variance only ('exp_var'), only based on corellation
            with the data in self.y ('corr_coef'), or are given in a list based
            on the exlained variance order ('list'). Default is 'exp_var'.
        **kwargs :
            All **kwargs from sklearn.decomposition.PCA are allowed, see the
            documentation of that class for details.
        **kwargs if mode=='exp_var':
            n_components : int
                The number of principal components used for the regression.
        **kwargs if mode=='corr_coef':
            n_components : int
                See above.
            max_components : int
                The number of components ordered by increasing explained
                variance which are used for calculation of corellation
                coefficients. Default is 10.
        **kwargs if mode=='list'
            used_pcs : list of int
                Gives the principal components used for regression based on the
                order of explained variance.

        Returns
        -------
        None

        """
        # PCA options
        copy = kwargs.get('copy', True)
        whiten = kwargs.get('whiten', False)
        svd_solver = kwargs.get('svd_solver', 'auto')
        tol = kwargs.get('tol', 0.0)
        iterated_power = kwargs.get('iterated_power', 'auto')
        random_state = kwargs.get('random_state', None)

        # # LinearRegression options
        # fit_intercept = kwargs.get('fit_intercept', True)
        # normalize = kwargs.get('normalize', False)
        # copy_X = kwargs.get('copy_X', True)
        # n_jobs = kwargs.get('n_jobs', None)

        if (mode == 'exp_var') or (mode == 'corr_coef'):
            n_components = kwargs.get('n_components')
        elif mode == 'list':
            pc_list = kwargs.get('used_pcs')
            n_components = max(pc_list)
        else:
            raise ValueError('Missing info on the number of components.')

        # Perform the actual PCA
        self.perform_pca(n_components, copy=copy, whiten=whiten,
                         svd_solver=svd_solver, tol=tol,
                         iterated_power=iterated_power,
                         random_state=random_state)

        # The number of components used for the calulation of the correlation
        # coefficients can be bigger than the number of components used for
        # the regression afterwards. Thus, e.g. a regression using 5 components
        # can include e.g. the 10th component based on the explained variance
        # ordering if max_components allows it.
        if mode == 'corr_coef':
            max_components = kwargs.get('max_components', 10)
            self.perform_pca(max_components, copy=copy, whiten=whiten,
                             svd_solver=svd_solver, tol=tol,
                             iterated_power=iterated_power,
                             random_state=random_state)

        # In each iteration in the following for loop, a multilinear regression
        # with the scores and only one of the responses given is performed. So
        # the loop iterates over the individual responses. This results in
        # one call of the fit and predict methods from the linear model objects
        # per iteration. This is necessary for mode=='coord_coef' because for
        # each response, different components may be used for regression.
        for curr_y in self.y_names:
            for curr_index, curr_scores in enumerate(
                    self.pca_scores.values.T):
                self.pcr_corr_coef.at[curr_index+1, curr_y] = np.abs(
                    np.corrcoef(
                        curr_scores, self.y[curr_y])[0, 1])

            # Select score values to be used for the regression based on the
            # mode given.
            if mode == 'exp_var':
                self.pcr_used_pcs.at[n_components, curr_y] = np.arange(
                    1, n_components+1)
            elif mode == 'corr_coef':
                self.pcr_used_pcs.at[n_components, curr_y] = np.flip(
                    np.argsort(self.pcr_corr_coef.loc[:, curr_y].values)[
                        -n_components:])+1
            elif mode == 'list':
                self.pcr_used_pcs.at[n_components, curr_y] = np.array(pc_list)

            fit_scores = self.pca_scores.loc[
                :, self.pcr_used_pcs.at[n_components, curr_y]].values

            # Build linear model
            fit_scores = sm.add_constant(fit_scores)
            self.pcr_models.at[n_components, curr_y] = sm.OLS(
                self.y[curr_y], fit_scores)
            self.pcr_params.at[n_components, curr_y] = (
                self.pcr_models.at[n_components, curr_y].fit())
            # Predict sample values according to PCR model
            idx = pd.IndexSlice
            self.pcr_y_c.loc[idx[curr_y, :], n_components] = (
                self.pcr_models.at[n_components, curr_y].predict(
                    self.pcr_params.at[n_components, curr_y].params,
                    fit_scores))
            # Cross-validate the PCR model
            self.pcr_y_cv.loc[idx[curr_y, :], n_components] = (
                cross_val_predict(statsmodel_wrapper(sm.OLS),
                                  fit_scores, self.y[curr_y],
                                  cv=round(100/cv_percentage)))
            # Calculate metrics for PCR model
            self.pcr_metrics.at[(curr_y, 'r2_c'), n_components] = r2_score(
                self.y[curr_y], self.pcr_y_c.loc[
                    idx[curr_y, :], n_components], multioutput='raw_values')
            self.pcr_metrics.at[(curr_y, 'r2_cv'), n_components] = r2_score(
                self.y[curr_y], self.pcr_y_cv.loc[
                    idx[curr_y, :], n_components], multioutput='raw_values')
            self.pcr_metrics.at[(curr_y, 'rmse_c'), n_components] = (
                mean_squared_error(self.y[curr_y],
                                   self.pcr_y_c.loc[idx[curr_y, :],
                                                    n_components],
                                   multioutput='raw_values', squared=False))
            self.pcr_metrics.at[(curr_y, 'rmse_cv'), n_components] = (
                mean_squared_error(self.y[curr_y],
                                   self.pcr_y_cv.loc[idx[curr_y, :],
                                                     n_components],
                                   multioutput='raw_values', squared=False))

    def pcr_sweep(self, sweep_components=20, cv_percentage=20, mode='exp_var',
                  **kwargs):
        """
        Perform PCR for all number of components between 1 and n_components.

        Parameters
        ----------
        sweep_components : int, optional
            The upper limit of components used for PCR. The default is 20.
        cv_percentage : float, optional
            Percentage of data used for cross-validation. The default is 10.
        mode : str, optional
            Determines if PCs used for regression are selected based on
            explained variance only ('exp_var'), only based on corellation
            with the data in self.y ('corr_coef'), or are given in a list based
            on the explained variance order ('list'). Default is 'exp_var'.
        **kwargs :
            The same **kwargs as in self.pcr_fit, only n_components must be
            left out because that is controlled by the iterator of the loop
            that repeatately calls self.pcr_fit.

        Returns
        -------
        DataFrame
            See Docstring of self.pcr_fit.

        """
        assert mode in ['exp_var', 'corr_coef'], (
            'No valid option for mode given. Remember that \'list\' is not '
            'allowed for this method because in this case information about '
            'the PCs used must already be known.')

        for ii in reversed(range(1, sweep_components+1)):
            self.pcr_fit(cv_percentage=cv_percentage, mode=mode,
                         n_components=ii, **kwargs)
        # return self.pcr_results

    def predict(self, samples, n_components, sample_names=None):
        """
        Predict unknown sample target values.

        Currently works correct only for mode=='exp_var'. For
        mode=='corr_coef', the tranformation has to be implemented independent
        of the PCA object method.

        Parameters
        ----------
        samples : ndarray
            Sample data in the shape (n_samples, n_variables).
        n_components : int
            Number of components used in the PCR model for the prediction.
        sample_names : list of str, optional
            The names of the samples passed to the function for prediction.
            Default is None resulting in numbered sample names.

        Returns
        -------
        prediction : ndarray
            Predicted target values in the shape (n_samples,) for a single
            target or (n_samples, n_targets) for multiple targets.

        """
        if sample_names is None:
            sample_names = ['Sample {}'.format(curr_idx)
                            for curr_idx in range(len(samples))]

        transformed_samples = self.pca_objects.at[n_components].transform(
            self.scaler.transform(samples))

        transformed_samples = sm.add_constant(transformed_samples)
        self.test = transformed_samples

        prediction = pd.DataFrame([], columns=self.y_names, index=sample_names)
        for curr_y in self.y_names:
            prediction[curr_y] = (
                self.pcr_models.at[n_components, curr_y].predict(
                    self.pcr_params.at[n_components, curr_y].params,
                    transformed_samples))
        return prediction

    def results_to_csv(self, folder=None):
        if folder is None:
            folder = ''
        self.pca_loadings.to_csv(folder + '/pca_loadings.txt', sep='\t')
        self.pca_explained_variance.to_csv(
            folder + '/pca_explained_variance.txt', sep='\t')
        self.pca_scores.to_csv(folder + '/pca_scores.txt', sep='\t')
        self.pcr_corr_coef.to_csv(folder + '/pcr_corr_coef.txt', sep='\t')
        self.pcr_metrics.to_csv(folder + '/pcr_metrics.txt', sep='\t')
        self.pcr_used_pcs.to_csv(folder + '/pcr_used_pcs.txt', sep='\t')
        self.pcr_y_c.to_csv(folder + '/pcr_y_c.txt', sep='\t')
        self.pcr_y_cv.to_csv(folder + '/pcr_y_cv.txt', sep='\t')

    def pca_biplot(self, pc_numbers=[1, 2], grouping=[None, None, None],
                   scores_only=False, **kwargs):
        """
        Make a plot containing both the PCA loadings and scores.

        Parameters
        ----------
        pc_numbers : list of int, optional, optional
            Contains two entries giving the component indices to be used for
            plotting, starting with 1 for the first principal component. The
            default is [0,1].
        grouping : list of str or None, optional
            Contains three elements that must either be an element from
            self.y_names or None. Defines a color (first entry), symbol
            (second entry) and symbol fill style grouping (third entry) of
            plotted scores based on the given targets. Default is [None, None,
            None] where there is no grouping and only one color and one symbol
            is used.
        scores_only : boolean, optional
            If True, the loadings will not be plotted, but only the scores.
            Default is False.
        **kwargs :
            colors : list of matplotlib color codes
                Must contain as many elements as levels present in the target
                used for the color grouping. By default, the colors are given
                by the matplotlib rainbow colormap.
            markers : list og matplotlib markers, optional
                Must contain as many elements as levels present in the target
                used for symbol grouping. By default, a modified list obtained
                by matplotlib.lines.Line2D.markers.keys() is used.
            fill_styles : list of matplotlib marker fill styles, optional
                Must contain as many elements as levels present in the target
                used for fill style grouping. By default, a modified list
                obtained by matplotlib.lines.Line2D.fillStyles is used.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Define plot deltails
        SMALL_SIZE = 6
        MEDIUM_SIZE = 8
        BIGGER_SIZE = 10
        FIGSIZE = (3.1496, 2.3622)
        DPI = 600
        MARKERSIZE = 4

        plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        # First produce axis with two circles and the coordinate axes as dotted
        # lines. This axis will be used to plot the PCA loadings.
        fig1, ax1 = plt.subplots(1, figsize=FIGSIZE, dpi=DPI)
        if not scores_only:
            circle1 = plt.Circle((0, 0), 1, color='k', linestyle='--',
                                 fill=False)
            circle2 = plt.Circle((0, 0), np.sqrt(0.5), color='k',
                                 linestyle='dotted', fill=False)
            ax1.add_artist(circle1)
            ax1.add_artist(circle2)
            ax1.axhline(0, color='k', linestyle='dotted')
            ax1.axvline(0, color='k', linestyle='dotted')
            # Draw arrows starting at the origin for the PCA loadings and
            # annotate them with the name of the corresponding target from
            # self.y.
            for (curr_loading1, curr_loading2, curr_var) in zip(
                    self.pca_loadings[pc_numbers[0]],
                    self.pca_loadings[pc_numbers[1]],
                    self.pca_loadings.index.values):
                ax1.arrow(0, 0, curr_loading1, curr_loading2, color='b',
                          head_width=0.05, length_includes_head=True)
                if curr_loading1 >= 0:
                    ha = 'left'
                else:
                    ha = 'right'
                # rotation = np.arctan(curr_loading2/curr_loading1)/(2*np.pi)*360
                ax1.text(curr_loading1, curr_loading2, curr_var, ha=ha,
                         va='center', rotation=0)
            # Set limits and labels
            ax1.set_xlim(-1.1, 1.1)
            ax1.set_ylim(-1.1, 1.1)
            ax1.set_xlabel('PC{} loadings'.format(pc_numbers[0]))
            ax1.set_ylabel('PC{} loadings'.format(pc_numbers[1]))
            ax1.set_facecolor('lightgrey')

            # Set x and y tick spacing
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.4))
            ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
            ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

        # Add another axis on top of the previous one. The new axis will hold
        # the PCA scores.
        ax2 = fig1.add_subplot(111)

        # If no grouping by color and/or symbol grouping contains only None
        # so a simple scatter plot is made for the scores.
        if not any(grouping):
            ax2.plot(self.pca_scores[pc_numbers[0]],
                     self.pca_scores[pc_numbers[1]],
                     label='Scores', marker='o',
                     markersize=MARKERSIZE, linestyle='none')
            ax2.legend(loc='center right', bbox_to_anchor=(1.7, 0.5))

        # If grouping by color and/or symbol the plotting procedure is a
        # little more complicated, involves three nested for loops to
        # accomodate the three grouping principles.
        elif ((grouping[0] in self.y_names) or (grouping[0] is None)) and (
                (grouping[1] in self.y_names) or (grouping[1] is None)) and (
                    (grouping[2] in self.y_names) or (grouping[2] is None)):

            # The values present in the y variables used for grouping are
            # determined.
            grouping_levels = []
            grouping_levels.append(
                self.y[grouping[0]].unique() if grouping[0] is not None
                else None)
            grouping_levels.append(
                self.y[grouping[1]].unique() if grouping[1] is not None
                else None)
            grouping_levels.append(
                self.y[grouping[2]].unique() if grouping[2] is not None
                else None)

            # Masks used for the later selection of data to be plotted are
            # determined. This is done for one y variable used for grouping
            # after the other (next three if-else statements).
            grouping_masks = []
            if grouping_levels[0] is not None:
                grouping_masks.append([
                    (self.y[grouping[0]] == curr_level).values
                    for curr_level in grouping_levels[0]])
            else:
                grouping_masks.append(
                    [np.full_like(self.y.index.values, True,
                                  dtype='bool')])

            if grouping_levels[1] is not None:
                grouping_masks.append([
                    (self.y[grouping[1]] == curr_level).values
                    for curr_level in grouping_levels[1]])
            else:
                grouping_masks.append([np.full_like(
                    self.y.index.values, True, dtype='bool')])

            if grouping_levels[2] is not None:
                grouping_masks.append([
                    (self.y[grouping[2]] == curr_level).values
                    for curr_level in grouping_levels[2]])
            else:
                grouping_masks.append([np.full_like(
                    self.y.index.values, True, dtype='bool')])

            # The 1D grouping masks from before are mixed together in one 3D
            # array that is iterated over while plotting later on.
            grouping_masks_all = (
                np.array(grouping_masks[0]) *
                np.array(grouping_masks[1])[:, np.newaxis]) * np.array(
                    grouping_masks[2])[:, np.newaxis, np.newaxis]

            # Colors, symbols and fill styles used for grouping are determined.
            grouping_colors = kwargs.get(
                'colors',
                [rainbow(x) for x in np.linspace(0, 1,
                                                 len(grouping_masks_all))])
            grouping_symbols = kwargs.get(
                'markers',
                ['o', 'v', 'P', 'X', 'H', '<', '>', '1', '2', '3', '4',
                 '8', 's', 'p', '*', 'h', '^', '+', 'x', 'D', 'd', '|',
                 '_', '.', ',', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            grouping_fillstyles = kwargs.get(
                'fill_styles',
                ['full', 'none', 'top', 'bottom', 'left', 'right'])

            # Now the actual plotting starts by iterating through the
            # previously calculated 3D mask array with three nested for loops.
            idx = np.empty(3, dtype='int')
            for idx[2], (curr_mask_2, curr_fillstyle) in enumerate(zip(
                    grouping_masks_all, grouping_fillstyles)):
                for idx[1], (curr_mask_1, curr_symbol) in enumerate(zip(
                        curr_mask_2, grouping_symbols)):
                    for idx[0], (curr_mask_0, curr_color) in enumerate(zip(
                            curr_mask_1, grouping_colors)):

                        label = ''
                        for ii, curr_levels in enumerate(grouping_levels):
                            if curr_levels is not None:
                                label = label + str(curr_levels[idx[ii]]) + ', '
                        label = label[:-2]
                        # if grouping_levels[0] is None:
                        #     label = str(grouping_levels[1][idx_0])
                        # elif grouping_levels[1] is None:
                        #     label = str(grouping_levels[0][idx_1])
                        # elif grouping_levels[2] is None:
                        #     label = str(grouping_levels[0][idx_2])
                        # else:
                        #     label = str(grouping_levels[0][idx_1])+', '+str(grouping_levels[1][idx_0])
    
                        ax2.plot(self.pca_scores.loc[curr_mask_0, pc_numbers[0]],
                                 self.pca_scores.loc[curr_mask_0, pc_numbers[1]],
                                 color=curr_color, marker=curr_symbol, label=label,
                                 fillstyle=curr_fillstyle, linestyle='none',
                                 markersize=MARKERSIZE)

            legend_title = ''
            for idx, curr_levels in enumerate(grouping_levels):
                if curr_levels is not None:
                    legend_title = legend_title + grouping[idx] + ', '
            legend_title = legend_title[:-2]
            ax2.legend(loc='center right', bbox_to_anchor= (1.8, 0.5), title=legend_title)
        else:
            raise ValueError('No valid grouping. Allowed values must be in'
                             ' {} or None, but \'{}\' was given.'.format(
                                 self.y_names, grouping))

        # Score plot axis labels and positions are fixed and the top layer
        # background is set to tranparent.
        ax2.yaxis.set_ticks_position('right')
        ax2.xaxis.set_ticks_position('top')
        ax2.yaxis.set_label_position('right')
        ax2.xaxis.set_label_position('top')
        ax2.set_xlabel('PC{} scores'.format(pc_numbers[0]))
        ax2.set_ylabel('PC{} scores'.format(pc_numbers[1]))
        ax2.patch.set_alpha(0)

        return fig1

    def generate_plots(self, plot_names, response_number=0, **kwargs):
        """
        Generate some basic plots of principal component regression results.

        Parameters
        ----------
        plot_names : list of str
            List of plots to be generated. Allowed entries are
            'actual_vs_pred' (actual target values vs. predicted values),
            'r2_vs_comp' (coefficient of determination vs. number of
            components), 'mse_vs_comp' (mean squared error vs. number of
            components), 'biplot' (PCA loadings and scores in one plot),
            'scoreplot_3D' (a 3D score plot using three principal components).
        response_number : int, optional
            Defines the index of the response from self.y to be plotted.
            Default is 0.
        **kwargs :
            n_components : int
                Needed for plot_name 'actual_vs_pred'. Currently, there is one
                difficulty when using mode=='list' for self.pcr_fit. In this
                case, the value for n_components has to be the highest number
                in the list, otherwise a KeyError occurs.
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
        idx = pd.IndexSlice
        if 'actual_vs_pred' in plot_names:
            n_components = kwargs.get('n_components')
            curr_y = self.y[self.y_names[response_number]]
            curr_c = self.pcr_y_c.loc[idx[self.y_names[response_number], :],
                                      n_components]
            curr_cv = self.pcr_y_cv.loc[idx[self.y_names[response_number], :],
                                        n_components]
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
                    self.pcr_metrics.at[(self.y_names[response_number],
                                         'r2_cv'), n_components]))
                plt.xlabel('Measured')
                plt.ylabel('Predicted')
            plots.append(fig1)
        if 'r2_vs_comp' in plot_names:
            curr_c_r2 = self.pcr_metrics.loc[(self.y_names[response_number],
                                              'r2_c')]
            curr_cv_r2 = self.pcr_metrics.loc[(self.y_names[response_number],
                                               'r2_cv')]

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
            curr_c_rmse = self.pcr_metrics.loc[(self.y_names[response_number],
                                                'rmse_c')]
            curr_cv_rmse = self.pcr_metrics.loc[(self.y_names[response_number],
                                                 'rmse_cv')]

            with plt.style.context(('ggplot')):
                fig3, ax3 = plt.subplots(figsize=(9, 5))
                ax3.plot(
                    curr_c_rmse, linestyle='--', marker='o')
                if plot_cv_data:
                    ax3.plot(
                        curr_cv_rmse, linestyle='--', marker='o')
                plt.ylabel('RMSE')
                plt.xlabel('Number of components')
            plots.append(fig3)
        if 'expvar_vs_corrcoef' in plot_names:
            curr_expvar = self.pca_explained_variance['each']
            curr_corrcoef = self.pcr_corr_coef[self.y_names[response_number]]

            with plt.style.context(('ggplot')):
                fig4, ax4 = plt.subplots(figsize=(9, 5))
                ax4.scatter(
                    curr_expvar, curr_corrcoef)
                for ii, txt in enumerate(curr_expvar.index):
                    ax4.annotate(txt, (curr_expvar[ii+1], curr_corrcoef[ii+1]))
                ax4.set_yscale('log')
                ax4.set_xscale('log')
                plt.ylabel('Correlation coefficients')
                plt.xlabel('Explained variance')
                plt.title(self.y_names[response_number])
            plots.append(fig4)
        if 'scoreplot_3D' in plot_names:
            pc_numbers = kwargs.get('pc_numbers', [0, 1, 2])
            grouping = kwargs.get('grouping', [None, None])
            fig1 = plt.figure()
            ax2 = Axes3D(fig1)
            marker_size = 200
            # If no grouping by color and/or symbol grouping contains only None
            # so a simple scatter plot is made for the scores.
            if not any(grouping):
                ax2.scatter(self.pca_scores[pc_numbers[0]],
                            self.pca_scores[pc_numbers[1]],
                            self.pca_scores[pc_numbers[2]],
                            label='Scores', s=marker_size)
                ax2.legend(loc='lower left', bbox_to_anchor= (1.1, 0))
            # If grouping by color and/or symbol the plotting procedure is a
            # little more complicated, involves two nested for loops to
            # accomodate the two grouping principles.
            elif ((grouping[0] in self.y_names) or (grouping[0] is None)) and (
                    (grouping[1] in self.y_names) or (grouping[1] is None)):
                grouping_levels_0 = self.y[grouping[0]].unique() if grouping[0] is not None else None
                grouping_levels_1 = self.y[grouping[1]].unique() if grouping[1] is not None else None

                if grouping_levels_0 is not None:
                    grouping_masks_0 = [
                        (self.y[grouping[0]]==curr_level).values
                        for curr_level in grouping_levels_0]
                else:
                    grouping_masks_0 = [np.full_like(self.y.index.values, True, dtype='bool')]

                if grouping_levels_1 is not None:
                    grouping_masks_1 = [
                        (self.y[grouping[1]]==curr_level).values
                        for curr_level in grouping_levels_1]
                else:
                    grouping_masks_1 = [np.full_like(self.y.index.values, True, dtype='bool')]

                grouping_masks_all = np.array(grouping_masks_1) * np.array(grouping_masks_0)[:, np.newaxis]
                grouping_colors = kwargs.get(
                    'colors',
                    [rainbow(x) for x in np.linspace(0, 1,
                                                     len(grouping_masks_all))])
                #['k', 'b', 'r']
                grouping_symbols = kwargs.get(
                    'markers',
                    ['o', 'v', 'P', 'X', 'H', '<', '>', '1', '2', '3', '4',
                     '8', 's', 'p', '*', 'h', '^', '+', 'x', 'D', 'd', '|',
                     '_', '.', ',', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
                for idx_1, (curr_mask_1, curr_color) in enumerate(zip(
                        grouping_masks_all, grouping_colors)):
                    for idx_0, (curr_mask_0, curr_symbol) in enumerate(zip(
                            curr_mask_1, grouping_symbols)):

                        if grouping_levels_0 is None:
                            label = str(grouping_levels_1[idx_0])
                            legend_title = grouping[1]
                        elif grouping_levels_1 is None:
                            label = str(grouping_levels_0[idx_1])
                            legend_title = grouping[0]
                        else:
                            label = str(grouping_levels_0[idx_1])+', '+str(grouping_levels_1[idx_0])
                            legend_title = grouping[0]+', '+grouping[1]

                        ax2.scatter(self.pca_scores.loc[curr_mask_0, pc_numbers[0]],
                                    self.pca_scores.loc[curr_mask_0, pc_numbers[1]],
                                    self.pca_scores.loc[curr_mask_0, pc_numbers[2]],
                                    color=curr_color, marker=curr_symbol,
                                    s=marker_size, label=label)

                        ax2.legend(loc='upper left', bbox_to_anchor= (1.15, 1), title=legend_title)
            else:
                raise ValueError('No valid grouping. Allowed values must be in'
                                 ' {} or None, but \'{}\' was given.'.format(
                                     self.y_names, grouping))
            ax2.yaxis.set_ticks_position('top')
            ax2.xaxis.set_ticks_position('top')
            ax2.yaxis.set_label_position('top')
            ax2.xaxis.set_label_position('top')
            ax2.set_xlabel('PC{} scores'.format(pc_numbers[0]))
            ax2.set_ylabel('PC{} scores'.format(pc_numbers[1]))
            ax2.set_zlabel('PC{} scores'.format(pc_numbers[2]))
            ax2.patch.set_alpha(0)

        return plots


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
