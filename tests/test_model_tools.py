# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 19:55:00 2021

@author: Alexander Southan
"""

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import unittest

from src.pyDataFitting.model_tools import model_tools


class TestModelTools(unittest.TestCase):

    def test_model_tools(self):
        # some made up, very simple data for the ols fitting in a DataFrame
        # with column names that are the factors and the response used later
        x, y = np.meshgrid(np.linspace(1, 20, 5), np.linspace(1, 20, 5))
        z = x + y
        noise = np.random.normal(size=z.shape)
        data = pd.DataFrame(np.array(
            [x.ravel(), y.ravel(), (z + noise).ravel()]).T,
            columns=['A', 'B', 'Resp'])
        
        # the model_tools instance for a model taking into account two-factor
        # interactions
        model_t = model_tools('2fi', ['A', 'B'], response_name='Resp')
        
        # ols is called with the model_string method from model_t
        fit_model = ols(model_t.model_string(), data=data).fit()
        
        # The used model string and the fit summary are printed
        print('The model string is:\n', model_t.model_string(), '\n')
        print(fit_model.summary(), '\n')
        print('The DataFrame used for model string generation is:\n' ,
              model_t.param_combinations, '\n')
        
        # a plot is made to demonstrate the calc_model_value function
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z+noise)
        z_fit = np.empty_like(x.ravel())
        for idx, (curr_x, curr_y) in enumerate(zip(x.ravel(), y.ravel())):
            z_fit[idx] = model_t.calc_model_value(
                [curr_x, curr_y], fit_model.params)
        z_fit = z_fit.reshape(x.shape)
        ax.plot_surface(y, x, z_fit, alpha=0.5, color='b')
        
        # The rest demonstrates how check_hierarchy is used
        print('--------Hierarchy check demonstration--------')
        # Parameter A is disabled for model string generation
        model_t.param_combinations.at['A', 'mask'] = False
        print('The model string without A is:\n', model_t.model_string(), '\n')
        # check for model hierarchy
        print('The model string after hierarchy check is:\n',
              model_t.model_string(check_hierarchy=True), '\n')
        # The inclusion of A due to model hierarchy is noted in param_combinations.
        print('The inclusion of A due to model hierarchy is noted in param_combinations\n' ,
              model_t.param_combinations, '\n')
        # Another call of check_hierarchy overwrites the previous info because it uses
        # the mask column in param_combinations
        model_t.check_hierarchy()
        print('Lost info due to second call of check_hierarchy\n' ,
              model_t.param_combinations, '\n')
