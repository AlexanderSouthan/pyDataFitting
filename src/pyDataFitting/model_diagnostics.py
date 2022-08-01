#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:03:24 2022

@author: Alexander Southan
"""

import numpy as np
from scipy.stats import percentileofscore
from scipy.special import erfinv


def percentiles(data):
    percentiles = np.empty_like(data)

    for ii, curr_data in enumerate(data):
        percentiles[ii] = percentileofscore(data, curr_data, kind='mean')

    return percentiles


def theo_residual_percentiles(residuals):
    # Calculation with the probit function,
    # see https://en.wikipedia.org/wiki/Probit
    theo_resid_percentiles = np.sqrt(2)*erfinv(2*percentiles(residuals)/100-1)

    return theo_resid_percentiles


if __name__ == "__main__":

    n_data = 200
    x = np.linspace(0, 10, n_data)
    # y = x**2 + np.random.normal(0, 5, n_data)
    y = x**2 + (np.random.random(n_data)-0.5)*40

    from pyDataFitting.lmfit_nonlinear_regression import nonlinear_regression
    import matplotlib.pyplot as plt

    fit = nonlinear_regression({'a': 6}, {'a': [0, 10]}, lambda val, a: val**a, x, y)

    fig1, ax1 = plt.subplots()
    ax1.plot(x, y, ls='none', marker='o')
    ax1.plot(x, y + fit.residual, ls='-')
    
    fig2, ax2 = plt.subplots()
    ax2.plot(fit.residual, theo_residual_percentiles(fit.residual), ls='none', marker='o')