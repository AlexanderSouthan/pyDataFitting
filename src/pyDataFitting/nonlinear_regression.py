# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import (
    differential_evolution, least_squares, basinhopping, brute, shgo,
    dual_annealing)
from scipy import integrate
from scipy.stats import gaussian_kde

from little_helpers.math_functions import (
    gaussian, langmuir_isotherm, triangle, langmuir_isotherm_hydrogel,
    langmuir_comp, langmuir_comp_hydrogel, Herschel_Bulkley,
    cum_dist_normal_with_rise, conductivity_hydrogel)


#####################################
# 2D nonlinear regression
#####################################

def nonlinear_regression(x_values, y_values, function_type, alg='evo',
                         z_values=None, non_fit_par={}, **kwargs):
    """
    Do a nonlinear regression on a dataset.

    The objective function to be minimized is the sum of the squared residuals.

    Parameters
    ----------
    x_values : ndarray
        A 1D array containing the independent variable.
    y_values : ndarray
        A 1D array containing the dependent variable. Must contain the same
        number of elements like x_values. If z_values are provided, this is
        also an idependent variable.
    function_type : string
        The function to be fitted to the data. Must be a valid function type
        for calc_function below.
    alg : string, optional
        The algorithm used to minimize the sum of the squared residuals.
        Allowed values are in the following list:
        ['evo', 'lm', 'basinhopping', 'brute', 'shgo', 'dual_annealing']
        'evo' means differential evolution and 'lm' means Levenberg-Marquardt.
        The default is 'evo'.
    z_values : ndarray , optional
        If provided, must be a 1D array with the same number of elements like
        x_values. The fit is then performed as a fit on the 3D dataset with
        x_values and y_values as the independent variables. Default is None,
        meaning that a 2D fit is performed only with x_values as independent
        variable.
    non_fit_par : dict, optional
        A dictionary containing additional parameters used when calling
        fit_error (which are then passed on to calc_function) that are needed
        to calculate the dependent variable values. The given values can be
        understood as parameters that are fixed for the fit. Default is an
        empty dictionary. Currently, this functionality only works if the
        corresponding parameters are the last ones in functions from
        little_helpers.math_functions.
    **kwargs :
        weights : 'string' or None, optional
            Defines the method of weight calculation. Can be 'kde'
            (weights determined by the kernel density estimate) or
            'inverse_y' (weights are the inverse of y_values). Default is
            None, meaning that all weights are equal.
        for alg in ['evo', 'brute', 'shgo', 'dual_annealing']:
            boundaries : list of tuples
                A list with tuples containing two elements each, giving the
                upper and lower boundaries for each parameter.
        for alg == 'evo' or alg == 'dual_annealing':
            max_iter : int, optional
                The maximum numer of iterations used during optimization. The
                default is 1000.
        for alg == 'brute':
            grid_points : int, optional
                The number of grid points along an axis. Default is 20.
        for alg == 'lm' and alg == 'basinhopping':
            initial_guess : array_like
                Initial guess of the parameters to be found.
        for alg == 'basinhopping':
            n_iter : int, optional
                The number of iterations. The default is 100.

    Returns
    -------
    OptimizeResult
        The object containing the optimization result.

    """
    algs = ['evo', 'lm', 'basinhopping', 'brute', 'shgo', 'dual_annealing']

    weights = kwargs.get('weights', None)
    if weights is None:
        weights = np.ones_like(x_values)
    elif weights == 'kde':  # kernel density estimation
        bw_method = kwargs.get('bandwidth', None)
        kde = gaussian_kde(x_values, bw_method=bw_method)
        weights = 1/kde.evaluate(x_values)
    elif weights == 'inverse_y':
        weights = 1/y_values

    if alg == algs[0]:  # 'evo'
        boundaries = kwargs.get('boundaries', None)
        max_iter = kwargs.get('max_iter', 1000)
        return differential_evolution(
            fit_error, bounds=boundaries, args=(
                x_values, y_values, function_type, weights, 'sum_of_squares',
                z_values, non_fit_par), maxiter=max_iter)

    elif alg == algs[1]:  # 'lm'
        initial_guess = kwargs.get('initial_guess', None)
        return least_squares(
            fit_error, initial_guess,
            args=(x_values, y_values, function_type, weights, 'residuals',
                  z_values), method='lm')

    elif alg == algs[2]:  # 'basinhopping'
        initial_guess = kwargs.get('initial_guess', None)
        n_iter = kwargs.get('n_iter', 100)
        return basinhopping(fit_error, initial_guess, minimizer_kwargs={
            'args':(x_values, y_values, function_type, weights,
                    'sum_of_squares', z_values)}, niter=n_iter)

    elif alg == algs[3]:  # 'brute'
        boundaries = kwargs.get('boundaries', None)
        grid_points = kwargs.get('grid_points', 20)
        return brute(
            fit_error, ranges=boundaries, args=(
                x_values, y_values, function_type, weights, 'sum_of_squares',
                z_values), Ns=grid_points)

    elif alg == algs[4]:  # 'shgo'
        boundaries = kwargs.get('boundaries', None)
        return shgo(
            fit_error, bounds=boundaries, args=(
                x_values, y_values, function_type, weights, 'sum_of_squares',
                z_values))

    elif alg == algs[5]:  # 'dual_annealing'
        boundaries = kwargs.get('boundaries', None)
        max_iter = kwargs.get('max_iter', 1000)
        return dual_annealing(
            fit_error, bounds=boundaries, maxiter=max_iter, args=(
                x_values, y_values, function_type, weights, 'sum_of_squares',
                z_values))

    else:
        raise ValueError('No valid alg. Allowed values must be in {}.'.format(
            algs))

def fit_error(fit_par, x_values, y_values, function_type, weights,
              mode='sum_of_squares', z_values=None, non_fit_par={}):
    """
    The objective function to be minimized with nonlinear_regresssion.

    Parameters
    ----------
    fit_par : list
        A list containing the current values of the fit parameters.
    x_values : ndarray
        See docstring of nonlinear_regression.
    y_values : ndarray
        See docstring of nonlinear_regression.
    function_type : string
        See docstring of nonlinear_regression.
    weights : ndarray
        An array containing the weights of the different data points. Must
        contain as many elements as x_values.
    mode : string, optional
        Determines if the sum of squares of the residuals is returned
        ('sum_of_squares') or the residuals themselves ('residuals'). The
        default is 'sum_of_squares'.
    z_values : ndarray , optional
        If provided, must be a 1D array with the same number of elements like
        x_values. The fit is then performed as a fit on the 3D dataset with
        x_values and y_values as the independent variables. Default is None,
        meaning that a 2D fit is performed only with x_values as independent
        variable.
    non_fit_par : dict, optional
        A dictionary containing additional parameters used when calling
        calc_function that are needed to calculate the dependent variable
        values. The given values can be understood as parameters that are
        fixed for the fit. Default is an empty dictionary.

    Returns
    -------
    float or ndarray
        The sum of squared residuals or the residuals, dpeneding on mode.

    """
    if z_values is None:
        second_independent = None
        to_be_fitted = y_values
    else:
        second_independent = y_values
        to_be_fitted = z_values

    curr_values = calc_function(x_values, fit_par, function_type,
                                y_values=second_independent, non_fit_par=non_fit_par)

    modes = ['sum_of_squares', 'residuals']
    if mode == modes[0]:  # sum_of_squares
        return np.sum(weights*(curr_values - to_be_fitted)**2)
    elif mode == modes[1]:  # residuals
        return weights*(curr_values - to_be_fitted)
    else:
        raise ValueError(
            'No valid fit error mode given, allowed values are {}.'.format(
                modes))


def calc_function(x_values, parameters, function_type, y_values=None,
                  non_fit_par={}):
    function_names = ['polynomial', 'Gauss', 'rectangle_gauss_convolution',
                      'Langmuir', 'triangle', 'power_law', 'exp_growth',
                      'Langmuir_hydrogel', 'Herschel_Bulkley',
                      'cum_dist_normal_with_rise']

    function_names_3D = ['quadratic_3D','langmuir_comp',
                         'langmuir_comp_hydrogel', 'conductivity_hydrogel']

    if function_type in function_names:
        # 'polynomial': order of parameters: [0]+[1]*x+[2]*x^2+[3]*x^3+...
        if function_type == function_names[0]:
            function_values = np.polynomial.polynomial.polyval(x_values,
                                                               parameters)

        # 'Gauss': order of parameters: amp, xOffset, yOffset, sigma [can be
        # repeated for superposition]
        elif function_type == function_names[1]:
            parameters = np.array(parameters).reshape(-1, 4)
            function_values = gaussian(
                x_values, parameters[:, 0], parameters[:, 1], parameters[:, 2],
                parameters[:, 3])

        # 'rectangle_gauss_convolution': order of parameters: amp, xOffset,
        # yOffset, sigma_Gauss, layer_thickness
        elif function_type == function_names[2]:
            x_spacing = np.abs(x_values[1]-x_values[0])
            x_min = x_values[0]
            x_max = x_values[-1]

            x_addition_datapoints = np.around(
                parameters[4]/(2*x_spacing)).astype(np.uint32)
            x_addition = x_addition_datapoints * x_spacing
            x_min_convolution = x_min - x_addition
            x_max_convolution = x_max + x_addition

            x_values_convolution = np.arange(
                x_min_convolution, x_max_convolution+x_spacing/2, x_spacing)

            y_gauss = parameters[0]/np.sqrt(2*np.pi*parameters[3]**2)*np.exp(
                -(x_values_convolution-parameters[1])**2/(2*parameters[3]**2))
            y_gauss_integral = integrate.cumtrapz(y_gauss, x_values_convolution,
                                                  initial=0)

            function_values = (y_gauss_integral[2*x_addition_datapoints:] -
                               y_gauss_integral[
                                   :len(x_values_convolution) -
                                   2*x_addition_datapoints] +
                               parameters[2])

        # 'Langmuir': order of parameters: qm, Ks
        elif function_type == function_names[3]:
            function_values = langmuir_isotherm(
                x_values, *parameters, **non_fit_par)

        elif function_type == function_names[4]:
            function_values = triangle(x_values, *parameters, **non_fit_par)

        elif function_type == function_names[5]:  # 'power_law'
            function_values = parameters[0]*(x_values**parameters[1])

        elif function_type == function_names[6]:  # 'exp_growth'
            function_values = parameters[0]*(1-np.exp(
                -parameters[1]*(x_values-parameters[2])))

        # 'Langmuir_hydrogel': order of parameters: q_m, K_s, phi_h2o, rho_hydrogel
        elif function_type == function_names[7]:
            function_values = langmuir_isotherm_hydrogel(
                x_values, *parameters, **non_fit_par)

        # 'Herschel_Bulkley': Order of parameters: yield_stress, k, n
        elif function_type == function_names[8]:
            function_values = Herschel_Bulkley(
                x_values, *parameters, **non_fit_par)

        # cum_dist_normal_with_rise: sigma, x_offset, slope, amp, linear_rise
        elif function_type == function_names[9]:
            function_values = cum_dist_normal_with_rise(
                x_values, *parameters, linear_rise='right', **non_fit_par)

    elif function_type in function_names_3D:
        if (y_values is None) or (len(x_values) != len(y_values)):
            raise ValueError('No valid values for second independent '
                             'dimension (y_values) given, must be an array '
                             'with equal length like x_values.')

        # 'quadratic_3D': order of parameters: [0]*x^2 + [1]*y^2 + [2]*x*y + [3]*x
        # + [4]*y + [5]
        if function_type == function_names_3D[0]:
            x_meshgrid, y_meshgrid = np.meshgrid(x_values, y_values)
            function_values = (
                parameters[0]*x_meshgrid**2 + parameters[1]*y_meshgrid**2 +
                parameters[2]*x_meshgrid*y_meshgrid +
                parameters[3]*x_meshgrid + parameters[4]*y_meshgrid +
                parameters[5])

        # 'langmuir_comp': order of parameters: qm, Ks1, Ks2
        elif function_type == function_names_3D[1]:
            function_values = langmuir_comp(x_values, y_values, *parameters, **non_fit_par)

        # 'langmuir_comp_hydrogel': order of parameters: qm, Ks1, Ks2, phi_h2o, rho_hydrogel
        elif function_type == function_names_3D[2]:
            function_values = langmuir_comp_hydrogel(x_values, y_values,
                                                     *parameters, **non_fit_par)

        # 'conductivity_hydrogel': oder of parameters: c_e_1, c_e_2, sigma0,
        # m1, m2, qm1, qm2, ks1, ks2, phi_h2o, rho_hydrogel
        elif function_type == function_names_3D[3]:
            function_values = conductivity_hydrogel(x_values, y_values,
                                                    *parameters, **non_fit_par)

    else:
        raise ValueError('Unknown function type, allowed '
                         'values are {} and {}.'.format(
                             function_names, function_names_3D))

    return function_values



#####################################
# 3D nonlinear regression
#####################################
# This should not be used for new code. The functionality is also available
# in the function nonlinear_regression above if you provide values for z_values

def nonlinear_regression_3D(x_values, y_values, z_values, function_type,
                            boundaries=None, initial_guess=None, max_iter=1000,
                            alg='evo'):
    assert alg in ['evo', 'lm'], ('No valid alg. Allowed values are \'%s\' and'
                                  ' \'%s\'' % ('evo', 'lm'))

    if alg == 'evo':
        return differential_evolution(fit_error_3D, bounds=boundaries,
                                      args=(x_values, y_values, z_values,
                                            function_type, alg),
                                      maxiter=max_iter)
    if alg == 'lm':
        return least_squares(fit_error_3D, initial_guess,
                             args=(x_values, y_values, z_values,
                                   function_type, alg), method='lm')


def fit_error_3D(fit_par, x_values, y_values, z_values, function_type,
                 alg='evo'):
    curr_values = calc_function_3D(x_values, y_values, fit_par, function_type)

    if alg == 'evo':
        return_value = np.sum((curr_values - z_values)**2)
    if alg == 'lm':  # still experimental
        return_value = curr_values - z_values
        return_value = np.sum(return_value, axis=0)
    return return_value


def calc_function_3D(x_values, y_values, parameters, function_type):

    function_names = ['quadratic_3D','langmuir_comp', 'langmuir_comp_hydrogel']
    assert function_type in function_names, 'Unknown function type.'

    # 'quadratic_3D': order of parameters: [0]*x^2 + [1]*y^2 + [2]*x*y + [3]*x
    # + [4]*y + [5]
    if function_type == function_names[0]:
        x_meshgrid, y_meshgrid = np.meshgrid(x_values, y_values)
        return (parameters[0]*x_meshgrid**2 + parameters[1]*y_meshgrid**2 +
                parameters[2]*x_meshgrid*y_meshgrid +
                parameters[3]*x_meshgrid + parameters[4]*y_meshgrid +
                parameters[5])

    # 'langmuir_comp': order of parameters: qm, Ks1, Ks2
    elif function_type == function_names[1]:
        return langmuir_comp(x_values, y_values, *parameters)
    # 'langmuir_comp_hydrogel': order of parameters: qm, Ks1, Ks2, phi_h2o, rho_hydrogel
    elif function_type == function_names[2]:
        return langmuir_comp_hydrogel(x_values, y_values, *parameters)
    else:
        raise ValueError('Unknown function type, allowed '
                         'values are {}'.format(function_names))
