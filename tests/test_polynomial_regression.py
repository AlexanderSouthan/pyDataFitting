#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:59:07 2021

@author: Alexander Southan
"""

import numpy as np
import unittest

from src.pyDataFitting import polynomial_regression


class TestPolynomialRegression(unittest.TestCase):

    def test_poly_reg(self):

        # test with evenly spaced x
        x = np.linspace(0, 10, 100)
        y = x**2
        y_fit, coefs = polynomial_regression.polynomial_fit(x, y, 2)

        self.assertAlmostEqual(coefs[0], 0, 5)
        self.assertAlmostEqual(coefs[1], 0, 5)
        self.assertAlmostEqual(coefs[2], 1, 5)

        # test with unevenly spaced, unsorted x
        x_rand = np.random.random(100)
        y_rand = 1.33 * x_rand**2 + 3*x_rand - 12
        y_fit_rand, coefs_rand = polynomial_regression.polynomial_fit(
            x_rand, y_rand, 2)

        self.assertAlmostEqual(coefs_rand[0], -12, 5)
        self.assertAlmostEqual(coefs_rand[1], 3, 5)
        self.assertAlmostEqual(coefs_rand[2], 1.33, 5)

        # test with evenly spaced x and point constraints
        x_constr = np.linspace(-1, 1, 351)
        y_constr = 32 * x_constr**3 - 33*x_constr**2 + 5
        y_fit_constr, coefs_constr = polynomial_regression.polynomial_fit(
            x_constr, y_constr, 6, fixed_points=[(0, 100), (-1, -100)])

        control_constr = [1.00000000e+02, -5.67136681e-02, -8.83085208e+02,
                          3.23403209e+01, 1.85941266e+03, -3.74395783e-01,
                          -1.14441824e+03]
        for curr_coef, curr_control in zip(coefs_constr, control_constr):
            self.assertAlmostEqual(curr_coef, curr_control, 3)

        # test with point constraints and unevenly spaced, unsorted x
        x_constr_rand = (np.random.random(351) - 0.5) * 2
        y_constr_rand = 32 * x_constr_rand**8 - 33*x_constr_rand**2 + 5
        y_fit_constr_rand, coefs_constr_rand = (
            polynomial_regression.polynomial_fit(
                x_constr_rand, y_constr_rand, 8,
                fixed_points=[(0, 100), (-1, -100)]))

        # The following tests are commented out because the result does not
        # seem to be numerically stable with a random x input. See remark in
        # the function docstring.

        # control = [100, -37.47991628, -1259.98249027, 395.13651343,
        #            4215.01259472, -1013.24008504, -5267.92261675,
        #            719.18471755, 2176.49374196]
        # for curr_coef, curr_control in zip(coefs_constr_rand, control):
        #     self.assertAlmostEqual(curr_coef, curr_control, 3)

        # test with point and slope constraints
        x_both_constr = np.linspace(-1, 1, 351)
        y_both_constr = 12*x_both_constr**3 - 13*x_both_constr**2 - 258
        y_fit_both_constr, coefs_both_constr = (
            polynomial_regression.polynomial_fit(
                x_both_constr, y_both_constr, 6,
                fixed_points=[(0, 100), (-1, -100)],
                fixed_slopes=[(0, 2300)]))

        control_both_constr = [100, 2300, -2495.42186186, -7629.25414864,
                               3833.93215887, 5502.01406169, -1365.75038395]
        for curr_coef, curr_control in zip(coefs_both_constr,
                                           control_both_constr):
            self.assertAlmostEqual(curr_coef, curr_control, 3)
