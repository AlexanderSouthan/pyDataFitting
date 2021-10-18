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

        x = np.linspace(0, 10, 100)
        y = x**2

        y_fit, coefs = polynomial_regression.polynomial_fit(x, y, 2)

        self.assertAlmostEqual(coefs[2], 1, 5)
