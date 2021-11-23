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
            x, y, 'polynomial', alg='evo',
            boundaries=[(-100, 100), (-100, 100)])
        lm_fit = nonlinear_regression(
            x, y, 'polynomial', alg='lm', initial_guess=[20, 30])
        basinhopping_fit = nonlinear_regression(
            x, y, 'polynomial', alg='basinhopping', initial_guess=[20, 30])
        brute_fit = nonlinear_regression(
            x, y, 'polynomial', alg='brute',
            boundaries=[(-100, 100), (-100, 100)],
            grid_points=20)
        shgo_fit = nonlinear_regression(
            x, y, 'polynomial', alg='shgo',
            boundaries=[(None, None), (None, None)])
        dualannealing_fit = nonlinear_regression(
            x, y, 'polynomial', alg='dual_annealing',
            boundaries=[(-100, 100), (-100, 100)], max_iter=1000)

        self.assertAlmostEqual(evo_fit.x[0], 10, 5)
        self.assertAlmostEqual(evo_fit.x[1], 5, 5)
        self.assertAlmostEqual(lm_fit.x[0], 10, 5)
        self.assertAlmostEqual(lm_fit.x[1], 5, 5)
        self.assertAlmostEqual(basinhopping_fit.x[0], 10, 5)
        self.assertAlmostEqual(basinhopping_fit.x[1], 5, 5)
        self.assertAlmostEqual(brute_fit[0], 10, 4)
        self.assertAlmostEqual(brute_fit[1], 5, 4)
        self.assertAlmostEqual(shgo_fit.x[0], 10, 5)
        self.assertAlmostEqual(shgo_fit.x[1], 5, 5)
        self.assertAlmostEqual(dualannealing_fit.x[0], 10, 5)
        self.assertAlmostEqual(dualannealing_fit.x[1], 5, 5)
