# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:34:07 2023

@author: southan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, zscore
from tqdm import tqdm
import statsmodels.api as sm

from pyDataFitting import nonlinear_regression, theo_residual_percentiles


# This example fits a noisy data row with a polynomial model, plots the data
# and the fit as well as the distribution of residuals. Additionally, it is
# demonstrated that the Shapiro Wilk test fails in about 5 % of the cases also
# for residuals obeying a normal distribution. Therefore, the calculations are
# repeated 100 times.

n_data = 2000
x = np.linspace(0, 10, n_data)

shapiro_results = []
shapiro_results_random = []
runs = 1
for ii in tqdm(range(runs)):
    # calculate y data with normal distributed noise
    y = x**2 + np.random.normal(0, 5, n_data)
    # calculate y data with uniformly distributed noise
    y_random = x**2 + np.random.random(n_data)*5

    # do the fits
    fit = nonlinear_regression({'a': 6}, {'a': [0, 10]}, lambda val, a: val**a, x, y)
    fit_random = nonlinear_regression({'a': 6}, {'a': [0, 10]}, lambda val, a: val**a, x, y_random)
    
    # perform Shapiro-Wilk tests
    shapiro_results.append(shapiro(fit.residual).pvalue)
    shapiro_results_random.append(shapiro(fit_random.residual).pvalue)

shapiro_failed = (np.array(shapiro_results) < 0.05).sum()
shapiro_failed_random = (np.array(shapiro_results_random) < 0.05).sum()
print('Shapiro-Wilk test failed in {}/{} for normal distributed residuals.'.format(shapiro_failed, runs))
print('Shapiro-Wilk test failed in {}/{} for uniformly distributed residuals.'.format(shapiro_failed_random, runs))

# plot data and fit
fig1, ax1 = plt.subplots()
ax1.plot(x, y, ls='none', marker='o')
ax1.plot(x, y + fit.residual, ls='-')

# calculate quantiles
theo_quantiles = theo_residual_percentiles(fit.residual)
theo_quantiles_random = theo_residual_percentiles(fit_random.residual)
sample_quantiles = zscore(fit.residual)
sample_quantiles_random = zscore(fit_random.residual)

# Make Q-Q plot with pyDataFitting
fig2, ax2 = plt.subplots()
ax2.plot(theo_quantiles, sample_quantiles, ls='none', marker='o', c='k', label='normal distribution')
ax2.plot(theo_quantiles_random, sample_quantiles_random, ls='none', marker='o', c='r', label='uniform distribution')
ax2.plot(theo_quantiles, theo_quantiles, c='b')
ax2.set_xlabel('Theoretical quantiles')
ax2.set_ylabel('Sample quantiles')
ax2.set_title('Q-Q plot with pyDataFitting')
ax2.legend()


# This is how the same result can be obtained with statsmodels
fig3, ax3 = plt.subplots()
probplot = sm.ProbPlot(sample_quantiles)
probplot_random = sm.ProbPlot(sample_quantiles_random)
qq = probplot.qqplot(ax=ax3, line='45', label='normal distribution', markerfacecolor='k', markeredgecolor='k')
qq_random = probplot_random.qqplot(ax=ax3, line='45', label='uniform distribution', markerfacecolor='r', markeredgecolor='r')
sm.qqline(qq.axes[0], line='45', fmt='b-')
ax3.set_title('Q-Q plot with statsmodels')
ax3.legend()
ax3.set_xlim(ax2.get_xlim())
ax3.set_ylim(ax2.get_ylim())