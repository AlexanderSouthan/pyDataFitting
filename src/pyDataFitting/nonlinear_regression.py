# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:29:12 2022

@author: aso
"""

import warnings
import numpy as np
import lmfit
from scipy.stats import gaussian_kde


def nonlinear_regression(initial_guess, bounds, func, x, y, z=None,
                         non_fit_par={}, alg='differential_evolution',
                         weights=None, **kwargs):
    """
    Do a nonlinear regression on a dataset.

    The objective function to be minimized is the sum of the squared residuals.

    Parameters
    ----------
    initial_guess : dict
        A dictionary with keys that are the names of the parameters passed to
        func. The values are the initial guesses for the fit. Not all methods
        use an initial guess, but the keys are important nevertheless to
        introduce the names of the fit parameters.
    bounds : dict
        A dictionary with keys that are the names of the parameters passed to
        func. The values are list with two entries, the first being the lower
        limit and the second being the upper limit of the fit parameters. Not
        all methods use boundaries, and they can also be -inf or inf.
    func: a callable function
        The function used to calculate the fit curve values. Must accept
        independent variables as the first arguments and additionally fit
        parameters as well as non-fit parameters.
    x : ndarray
        A 1D array containing the independent variable.
    y : ndarray
        A 1D array containing the dependent variable. Must contain the same
        number of elements like x. If z is provided, this is also an idependent
        variable.
    z : ndarray , optional
        If provided, must be a 1D array with the same number of elements like
        x. The fit is then performed as a fit on the 3D dataset with x and y as
        the independent variables. Default is None, meaning that a 2D fit is
        performed only with x as independent variable.
    non_fit_par : dict, optional
        A dictionary containing additional parameters used when calling
        func that are needed to calculate the dependent variable values. The
        given values can be understood as parameters that are fixed for the
        fit. Default is an empty dictionary.
    alg : string, optional
        The algorithm used to minimize the sum of the squared residuals.
        Allowed values are all methods possible for lmfit.minimize.
    weights : 'string' or None, optional
        Defines the method of weight calculation. Can be 'kde' (weights
        determined by the kernel density estimate) or 'inverse_y' (weights are
        the inverse of y_values). Default is None, meaning that all weights are
        equal.
    **kwargs :
        All **fit_kws possible for lmfit.minimize.

    Returns
    -------
    OptimizeResult
        The object containing the optimization result.

    """
    params = lmfit.Parameters()
    for curr_key, curr_guess in initial_guess.items():
        params.add(curr_key, curr_guess, min=bounds[curr_key][0],
                   max=bounds[curr_key][1])

    if weights is None:
        weights = np.ones_like(x)
    elif weights == 'kde':  # kernel density estimation
        bw_method = kwargs.get('bandwidth', None)
        kde = gaussian_kde(x, bw_method=bw_method)
        weights = 1/kde.evaluate(x)
    elif weights == 'inverse_y':
        weights = 1/y
    else:
        weights = np.ones_like(x)
        warnings.warn('Invalid value for weights given, so uniform weights '
                      'are used')

    result = lmfit.minimize(fit_error, params,
                            args=(func, x, y, weights, 'residuals', 
                                  z, non_fit_par), method=alg, **kwargs)
    return result

def fit_error(params, func, x, y, weights, mode='sum_of_squares',
              z=None, non_fit_par={}):
    """
    The objective function to be minimized with nonlinear_regresssion.

    Parameters
    ----------
    params : list
        An lmfit Parameters set containing the current values and boundaries
        of the fit parameters.
    x : ndarray
        See docstring of nonlinear_regression.
    y : ndarray
        See docstring of nonlinear_regression.
    func : string
        See docstring of nonlinear_regression.
    weights : ndarray
        An array containing the weights of the different data points. Must
        contain as many elements as x.
    mode : string, optional
        Determines if the sum of squares of the residuals is returned
        ('sum_of_squares') or the residuals themselves ('residuals'). The
        default is 'sum_of_squares'.
    z : ndarray , optional
        If provided, must be a 1D array with the same number of elements like
        x. The fit is then performed as a fit on the 3D dataset with
        x and y as the independent variables. Default is None,
        meaning that a 2D fit is performed only with x as independent
        variable.
    non_fit_par : dict, optional
        A dictionary containing additional parameters used when calling
        func that are needed to calculate the dependent variable
        values. The given values can be understood as parameters that are
        fixed for the fit. Default is an empty dictionary.

    Returns
    -------
    float or ndarray
        The sum of squared residuals or the residuals, depending on mode.

    """
    if z is None:
        indep = [x]
        dep = y
    else:
        indep = [x, y]
        dep = z

    curr_values = func(*indep, **params, **non_fit_par)

    modes = ['sum_of_squares', 'residuals']
    if mode == modes[0]:  # sum_of_squares
        return np.sum(weights*(curr_values - dep)**2)
    elif mode == modes[1]:  # residuals
        return weights*(curr_values - dep)
    else:
        raise ValueError(
            'No valid fit error mode given, allowed values are {}.'.format(
                modes))
