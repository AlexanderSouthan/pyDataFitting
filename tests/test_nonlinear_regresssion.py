# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 19:55:00 2021

@author: Alexander Southan
"""

import numpy as np
import unittest

from src.pyDataFitting.nonlinear_regression import nonlinear_regression


class TestNonlinearRegression(unittest.TestCase):

    def test_nonlinear_reg(self):
        x = np.linspace(-100, 100)
        y = np.polynomial.polynomial.polyval(x, [10, 5])

        evo_fit = nonlinear_regression(
            {'m': 0, 'b': 0},
            {'m': [-100, 100], 'b': [-100, 100]},
            self.polynomial, x, y, alg='differential_evolution')
        lm_fit = nonlinear_regression(
            {'m': 0, 'b': 0},
            {'m': [-100, 100], 'b': [-100, 100]},
            self.polynomial, x, y, alg='least_squares')
        basinhopping_fit = nonlinear_regression(
            {'m': 0, 'b': 0},
            {'m': [-100, 100], 'b': [-100, 100]},
            self.polynomial, x, y, alg='basinhopping')
        # brute_fit = nonlinear_regression(
        #     {'m': 0, 'b': 0},
        #     {'m': [-100, 100], 'b': [-100, 100]},
        #     self.polynomial, x, y, alg='brute')
        # shgo_fit = nonlinear_regression(
        #     {'m': 5, 'b': 5},
        #     {'m': [0, 20], 'b': [0, 20]},
        #     self.polynomial, x, y, alg='shgo')
        dualannealing_fit = nonlinear_regression(
            {'m': 0, 'b': 0},
            {'m': [-100, 100], 'b': [-100, 100]},
            self.polynomial, x, y, alg='dual_annealing')

        self.assertAlmostEqual(evo_fit.params['b'], 10, 5)
        self.assertAlmostEqual(evo_fit.params['m'], 5, 5)
        self.assertAlmostEqual(lm_fit.params['b'], 10, 5)
        self.assertAlmostEqual(lm_fit.params['m'], 5, 5)
        self.assertAlmostEqual(basinhopping_fit.params['b'], 10, 3)
        self.assertAlmostEqual(basinhopping_fit.params['m'], 5, 3)
        # self.assertAlmostEqual(brute_fit.params['b'], 10, 4)
        # self.assertAlmostEqual(brute_fit.params['m'], 5, 4)
        # self.assertAlmostEqual(shgo_fit.params['b'], 10, 5)
        # self.assertAlmostEqual(shgo_fit.params['m'], 5, 5)
        self.assertAlmostEqual(dualannealing_fit.params['b'], 10, 5)
        self.assertAlmostEqual(dualannealing_fit.params['m'], 5, 5)

    def polynomial(self, x, m, b):
        return np.polynomial.polynomial.polyval(x, [b, m])