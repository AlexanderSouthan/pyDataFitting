# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 21:26:39 2021

@author: Alexander Southan
"""

import numpy as np
import pandas as pd
from itertools import combinations, combinations_with_replacement


class model_tools:
    def __init__(self, model_type, param_names, param_types=None,
                 response_name='R'):
        """
        Initialize model_tools instance.

        Parameters
        ----------
        model_type : string
            The model type, must be an element of self.model_types, so
            currently ['linear', '2fi', '3fi', 'quadratic'].
        param_names : list of string
            The parameter names used in the model string.
        param_types : None or list of string, optional
            Defines the type of the parameters. Can be a list with as many
            entries as param_names. Allowed entries are 'cont' for continuous
            parameters and 'categ' for categorial parameters. The default is
            None, meaning that all parameters are continuous.
        response_name : string, optional
            The name of the response. The default is 'R'.

        Raises
        ------
        ValueError
            If invalid model_Type is given.

        Returns
        -------
        None.

        """
        # Careful when adding another model, all references to the following
        # list have to be updated
        self.model_types = ['linear', '2fi', '3fi', 'quadratic',
                            'quadratic+3fi']
        self.model_type = model_type
        self.param_names = np.asarray(param_names)
        if param_types is None:
            self.param_types = ['cont']*len(param_names)
        else:
            self.param_types = param_types
        self.response_name = response_name

        if self.model_type not in self.model_types:
            raise ValueError('No valid model_type given. Should be an element '
                             'of {}, but is \'{}\'.'.format(
                                 self.model_types, self.model_type))

        param_numbers = np.arange(len(self.param_names))

        self.param_combinations = pd.DataFrame(
            [], index=self.param_names, columns=np.append(
                self.param_names, ['string', 'mask', 'for_hierarchy']))

        # linear part is used for all models
        # for curr_name in self.param_names:
        combi_array = np.diag(np.ones_like(self.param_names, dtype='int'))
        self.param_combinations.loc[self.param_names, 'string'] = (
            self.param_names)

        # two-factor interactions
        if self.model_type in self.model_types[1:5]: #  '2fi', '3fi', 'quadratic', 'quadratic+3fi'
            simple_2fi = list(combinations(param_numbers, 2))
            for subset in simple_2fi:
                curr_idx = ':'.join(i for i in self.param_names[list(subset)])
                curr_string = '*'.join(i for i in self.param_names[list(subset)])
                self.param_combinations.at[curr_idx, 'string'] = curr_string

                curr_combi = np.zeros_like(self.param_names, dtype='int')
                curr_combi[list(subset)] = 1
                combi_array = np.append(combi_array, [curr_combi], axis=0)

        # quadratic 2fi terms
        if self.model_type in self.model_types[3:5]:  # 'quadratic', 'quadratic+3fi'
            all_2fi = list(combinations_with_replacement(param_numbers, 2))
            quadratic_mask = [True if curr_2fi not in simple_2fi else False
                              for curr_2fi in all_2fi]
            for subset in np.array(all_2fi)[quadratic_mask]:
                curr_idx = 'I({} * {})'.format(*self.param_names[subset])
                curr_string = 'I({}*{})'.format(*self.param_names[subset])
                self.param_combinations.at[curr_idx, 'string'] = curr_string

                curr_combi = np.zeros_like(self.param_names, dtype='int')
                curr_combi[list(subset)] = 2
                combi_array = np.append(combi_array, [curr_combi], axis=0)

            # curr_combi = np.diag(np.full_like(self.param_names, 2,
            #                                   dtype='int'))
            # combi_array = np.append(combi_array, curr_combi, axis=0)

        # three-factor interactions
        if self.model_type in [self.model_types[2], self.model_types[4]]:  # '3fi', 'quadratic+3fi'
            simple_3fi = list(combinations(param_numbers, 3))
            for subset in simple_3fi:
                curr_idx = ':'.join(i for i in self.param_names[list(subset)])
                curr_string = '*'.join(i for i in self.param_names[list(subset)])
                self.param_combinations.at[curr_idx, 'string'] = curr_string

                curr_combi = np.zeros_like(self.param_names, dtype='int')
                curr_combi[list(subset)] = 1
                combi_array = np.append(combi_array, [curr_combi], axis=0)

        # # quadratic 3fi terms
        # if self.model_type == self.model_types[4]:  # 'quadratic+3fi'
        #     all_3fi = list(combinations_with_replacement(param_numbers, 3))
        #     quadratic_mask = [True if ((curr_3fi not in simple_3fi) & (not np.all(curr_3fi[0]==curr_3fi))) else False
        #                       for curr_3fi in all_3fi]
        #     for subset in np.array(all_3fi)[quadratic_mask]:
        #         curr_idx = 'I({} * {} * {})'.format(*self.param_names[subset])
        #         curr_string = 'I({}*{}*{})'.format(*self.param_names[subset])
        #         self.param_combinations.at[curr_idx, 'string'] = curr_string

        #         curr_combi = np.zeros_like(self.param_names, dtype='int')
        #         curr_combi[list(subset)] = 1
        #         combi_array = np.append(combi_array, [curr_combi], axis=0)


        self.param_combinations[self.param_names] = combi_array
        self.param_combinations['mask'] = True
        self.param_combinations['for_hierarchy'] = False

        # Drop quadratic terms for categoric factors
        if self.model_type in self.model_types[3:5]:  # 'quadratic', 'quadratic+3fi'
            for curr_name, curr_type in zip(self.param_names,
                                            self.param_types):
                if curr_type == 'categ':
                    self.param_combinations.drop('I({} * {})'.format(
                        curr_name, curr_name), inplace=True, axis=0)

    def model_string(self, combi_mask=None, check_hierarchy=False):
        """
        Generate the model string necessary for OLS fitting.

        Parameters
        ----------
        combi_mask : pd.Series
            A series containing boolean values which define if certain
            parameter combinations are included into the model string. The
            index should be identical to the index of self.param_combinations.
            The default is None, meaning that all parameter combinations are
            included into the model string.
        check_hierarchy : bool, optional
            Defines if the model hierarchy is checked and corrected if
            necessary. The deafult is False.

        Returns
        -------
        model_string : string
            The model string in the correct format to be used by the OLS
            function in experimental_design.perform_anova.

        """
        if combi_mask is not None:
            self.param_combinations['mask'] = combi_mask
        if check_hierarchy:
            self.check_hierarchy()

        return '{} ~ {} + 1'.format(
            self.response_name, self.param_combinations.loc[
                self.param_combinations['mask'], 'string'].str.cat(sep=' + '))

    def check_hierarchy(self):
        """
        Check for hierarchy of the model implemented.

        All entries with False in self.param_combinations['mask'] are checked
        if they should be included in the model for hierarchy. If this is
        found for a parameter or a parameter combination, the corresponding
        entry in the DataFrame is set to True and the value in the column
        'for_hierarchy' is also set to True in order to show that this term is
        only included for hierarchy and not due to a significant contribution.

        Returns
        -------
        None.

        """
        self.param_combinations['for_hierarchy'] = False
        excluded_mask = ~self.param_combinations['mask']
        combi_data = self.param_combinations[self.param_names]
        check_data = self.param_combinations.loc[excluded_mask,
                                                 self.param_names]

        for curr_combi in check_data.index:
            curr_mask = check_data.loc[curr_combi]>0
            curr_hier = combi_data.where(combi_data.loc[:, curr_mask] > 0, 0)
            deps = curr_hier.merge(
                check_data.loc[[curr_combi]], indicator=True, how='left',
                on=curr_hier.columns.to_list())['_merge']=='both'
            deps.index = curr_hier.index
            if (deps*self.param_combinations['mask']).any():
                self.param_combinations.at[curr_combi, 'for_hierarchy'] = True
                self.param_combinations.at[curr_combi, 'mask'] = True

    def calc_front_factors(self, coded_values):
        """
        Calcualte the front factors of the individual model terms.

        Calculation is done for a set of coded values, i.e. values that are
        between -1 and 1. The result is useful for a quick calculation of
        model predictions using self.calc_model_values.

        Parameters
        ----------
        coded_values : list of int
            A list containing the coded values, i.e. values between -1 and 1
            are allowed. The list should contain as many elements as the model
            used for data analysis has.

        Returns
        -------
        front_factors : Series
            The front factors for the individual model terms. The index is the
            same like in the params property of the models, so the two Series
            can be used for calculations easily.

        """
        front_factors = pd.Series([1], index=['Intercept'], dtype='float')
        # # linear terms
        # for curr_value, curr_name in zip(coded_values, self.param_names):
        #     front_factors[curr_name] = curr_value
        # # two-factor interactions
        # if self.model_type in self.model_types[1:4]: #  '2fi', '3fi', 'quadratic'
        #     for subset_values, subset_names in zip(
        #             itertools.combinations(coded_values, 2),
        #             itertools.combinations(self.param_names, 2)):
        #         front_factors['{}'.format(
        #             ":".join(i for i in subset_names))] = np.prod(
        #                 subset_values)
        # # three-factor interactions
        # if self.model_type == self.model_types[2]:  # '3fi'
        #     for subset_values, subset_names in zip(
        #             itertools.combinations(coded_values, 3),
        #             itertools.combinations(self.param_names, 3)):
        #         front_factors['{}'.format(
        #             "*".join(i for i in subset_names))] = np.prod(
        #                 subset_values)
        # # quadratic terms
        # if self.model_type == self.model_types[3]:  # 'quadratic'
        #     for curr_value, curr_name in zip(coded_values, self.param_names):
        #         front_factors['{}:{}'.format(curr_name, curr_name)] = (
        #             curr_value**2)

        # The following code tries to take advantage of the calculations done
        # in the model_tools objects already, however is much slower than
        # repeating the calculations as done above. It is preferred to the code
        # above nontheless in order to avoid inconsistency.

        coded_values = np.asarray(coded_values)
        combi_matrix = self.param_combinations.loc[
            self.param_combinations['mask']==True, self.param_names]
        for curr_combi in combi_matrix.index:
            curr_mask = combi_matrix.loc[curr_combi].astype(bool)
            front_factors[curr_combi] = np.prod(
                coded_values[curr_mask]**combi_matrix.loc[curr_combi,
                                                          curr_mask])

        return front_factors

    def calc_model_value(self, param_values, model_coefs):
        """
        Calculate one value the model predicts.

        Calculation is done for one set of parameter settings and the model
        coefficients have to be provided.

        Parameters
        ----------
        param_values : list of float
            A list containing one set of parameter values. Must contain as many
            elements as there are active parameters in the model. Active
            parameters are those with True in self.param_combinations['mask'].
            The order is given by the order of the index of
            self.param_combinations.
        model_coefs : list of float
            A list containing the model coefficients, for example obtained by
            regression. Must contain the same numerb of elements like
            param_values, also in the same order.

        Returns
        -------
        float
            The predicted response value for the parameter settings.

        """
        model_coefs = np.asarray(model_coefs)
        front_factors = self.calc_front_factors(param_values)
        return (front_factors * model_coefs).sum()
