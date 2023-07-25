# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:34:07 2023

@author: southan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from tqdm import tqdm

from pyDataFitting import nonlinear_regression, theo_residual_percentiles


# This example fits a noisy data row with a polynomial model, plots the data
# and the fit as well as the distribution of residuals. Additionally, it is
# demonstrated that the Shapiro Wilk test fails in about 5 % of the cases also
# for residuals obeying a normal distribution. Therefore, the calculations are
# repeated 5000 times.

n_data = 200
x = np.linspace(0, 10, n_data)

shapiro_results = []
runs = 5000
for ii in tqdm(range(runs)):
    y = x**2 + np.random.normal(0, 5, n_data)
    # y = x**2 + (np.random.random(n_data)-0.5)*40

    fit = nonlinear_regression({'a': 6}, {'a': [0, 10]}, lambda val, a: val**a, x, y)
    shapiro_results.append(shapiro(fit.residual).pvalue)

shapiro_failed = (np.array(shapiro_results) < 0.05).sum()
print('Shapiro-Wilk test failed in {}/{}.'.format(shapiro_failed, runs))

fig1, ax1 = plt.subplots()
ax1.plot(x, y, ls='none', marker='o')
ax1.plot(x, y + fit.residual, ls='-')

fig2, ax2 = plt.subplots()
ax2.plot(fit.residual, theo_residual_percentiles(fit.residual), ls='none', marker='o')