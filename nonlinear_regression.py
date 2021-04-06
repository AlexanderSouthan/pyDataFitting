# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy import integrate
from scipy.stats import gaussian_kde


#####################################
# 2D nonlinear regression
#####################################

def nonlinear_regression(x_values, y_values, function_type, z_values=None,
                         boundaries=None, initial_guess=None, max_iter=1000,
                         alg='evo', weights=None, **kwargs):
    assert alg in ['evo', 'lm'], ('No valid alg. Allowed values are \'%s\' '
                                  'and \'%s\'') % ('evo', 'lm')

    if weights is None:
        weights = np.ones_like(x_values)
    elif weights == 'kde':  # kernel density estimation
        bw_method = kwargs.get('bandwidth', None)
        kde = gaussian_kde(x_values, bw_method=bw_method)
        weights = 1/kde.evaluate(x_values)

    if alg == 'evo':
        return differential_evolution(
            fit_error, bounds=boundaries,
            args=(x_values, y_values, function_type, weights, 'sum_of_squares'),
            maxiter=max_iter)
    if alg == 'lm':
        return least_squares(
            fit_error, initial_guess,
            args=(x_values, y_values, function_type, 'residuals'), method='lm')


def fit_error(fit_par, x_values, y_values, function_type, weights,
              mode='sum_of_squares'):
    curr_values = calc_function(x_values, fit_par, function_type)

    modes = ['sum_of_squares', 'residuals']
    if mode == modes[0]:  # sum_of_squares
        return np.sum(weights*(curr_values - y_values)**2)
    elif mode == modes[1]:  # residuals
        return weights*(curr_values - y_values)
    else:
        raise ValueError(
            'No valid fit error mode given, allowed values are {}.'.format(
                modes))


def calc_function(x_values, parameters, function_type):
    function_names = ['polynomial', 'Gauss', 'rectangle_gauss_convolution',
                      'Langmuir', 'triangle', 'power_law', 'exp_growth']
    # 'polynomial': order of parameters: [0]+[1]*x+[2]*x^2+[3]*x^3+...
    if function_type == function_names[0]:
        function_values = np.full_like(
            x_values, parameters[0], dtype='float64')
        for ii, curr_parameter in enumerate(parameters[1:]):
            function_values += curr_parameter * x_values**(ii+1)

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
            x_values, *parameters)

    elif function_type == function_names[4]:
        function_values = triangle(x_values, *parameters)

    elif function_type == function_names[5]:  # 'power_law'
        function_values = parameters[0]*(x_values**parameters[1])

    elif function_type == function_names[6]:  # 'exp_growth'
        function_values = parameters[0]*(1-np.exp(
            -parameters[1]*(x_values-parameters[2])))

    else:
        raise ValueError('Unknown function type, allowed '
                         'values are {}'.format(function_names))

    return function_values

def langmuir_isotherm(c_e, q_m, K_s):
    """
    Calculate the q_e values of a Langmuir isotherm.

    Parameters
    ----------
    c_e : ndarray
        The equilibrium concentrations in the liquid phase. Can have any shape,
        so an (M, N) array may be interpreted as M data rows with N data
        points.
    q_m : float
        The adsorption capacity of the adsorber.
    K_s : float
        The equilibrium constant of adsorption and desorption.

    Returns
    -------
    ndarray
        The equilibrium concentrations q_e in the adsorber. Has the same shape
        like c_e.

    """
    return q_m * c_e * K_s/(1 + c_e * K_s)

def triangle(x, start_left, start_right, x_max, y_max, y_offset=0):
    """
    Calculate a triangle function. 
    
    The triangle function is zero outside of the triangle and different slopes
    on both sides of the triangle are possible.

    Parameters
    ----------
    x : ndarray
        The x values used for the calculation. Can be any shape, but the
        triangle will always be produced in the last dimension. An (M, N) array
        can therefore be interpreted as M data rows with potentially different
        x values while the triangles are always created at the same x values.
    start_left : float
        The x value where the triangle starts, i.e. the left edge of the
        tiangle.
    start_right : float
        The x value where the triangle stops, i.e. the right edge of the
        triangle.
    x_max : float
        The x value of the triangle maximum/minimum, must be between start_left
        and start_right, otherwise odd results will occur.
    y_max : float
        The y value of the triangle maximum/minimum.
    y_offset : float, optional
        The y value outside of the triangle. Default is 0. If y_offset is
        greater than y_max, the triangle will point downwards, otherwise
        upwards.

    Returns
    -------
    triangle : ndarray
        An array containing the funtion values of the triangle functions. Has
        the same shape like x.

    """
    left_mask = np.logical_and(
        x >= min(start_left, x_max),
        x <= max(start_left, x_max))
    right_mask = np.logical_and(
        x > min(x_max, start_right),
        x <= max(x_max, start_right))

    left_slope = (y_max-y_offset)/(x_max - start_left)
    right_slope = -(y_max-y_offset)/(start_right - x_max)

    triangle = np.full_like(x, y_offset)
    triangle[left_mask] += left_slope * (x[left_mask] - start_left)
    triangle[right_mask] += right_slope * (x[right_mask] - start_right)
    return triangle

def gaussian(x, amp, x_offset, y_offset, sigma):
    """
    Calculate one or a superposition of Gaussian normal distributions.

    Parameters
    ----------
    x : ndarray
        A one-dimensional array with the x values used for calculations.
    amp : float or list of float
        The amplitudes, i.e. the maximum values of the calculated Gauss curve.
        If a single value is given, a single peak is created. If a list of
        values is given, a superposition of several Gauss curves will be
        calculated.
    x_offset : float or list of float
        The x position of the maximum value defined by amp. Must be the same
        shape like amp.
    y_offset : float or list of float
        The y value of the baseline of the Gauss curve. Must be the same shape
        like amp.
    sigma : float or list of float
        The with of the Gauss curve. The full width at half maximum is given by
        2*sqrt(2*ln(2))*sigma. Must be the same shape like amp.

    Returns
    -------
    ndarray
        An array containing the function values of the (superimposed) Gauss
        curves. Has the same shape like x.

    """
    amp = np.array(amp, ndmin=1)
    x_offset = np.array(x_offset, ndmin=1)
    y_offset = np.array(y_offset, ndmin=1)
    sigma = np.array(sigma, ndmin=1)
    return np.sum(
        amp[:, np.newaxis] * np.exp(
            (x - x_offset[:, np.newaxis])**2 /
            (-2 * sigma[:, np.newaxis]**2)) +
        y_offset[:, np.newaxis], axis=0)

def boxcar(x, boxcar_start, boxcar_end, y_offset=0, amp=1):
    boxcar_mask = np.logical_and(
        x >= min(boxcar_start, boxcar_end),
        x <= max(boxcar_start, boxcar_end))
    y_boxcar = np.full_like(x, y_offset)
    y_boxcar[boxcar_mask] = amp
    return y_boxcar

def boxcar_convolution(x, boxcar_start, boxcar_end, convolution_function,
                       con_func_params, y_offset=0):
    x_spacing = np.abs(x[1]-x[0])
    x_min = x[0]
    x_max = x[-1]
    boxcar_width = abs(boxcar_start - boxcar_end)

    x_addition_datapoints = np.around(
        boxcar_width/(2*x_spacing)).astype(np.uint32)
    x_addition = x_addition_datapoints * x_spacing
    x_min_convolution = x_min - x_addition
    x_max_convolution = x_max + x_addition

    x_values_convolution = np.arange(
        x_min_convolution, x_max_convolution+x_spacing/2, x_spacing)

    y_con_func = convolution_function(x_values_convolution, *con_func_params)
    y_con_func_integral = integrate.cumtrapz(y_con_func, x_values_convolution,
                                             initial=0)

    function_values = (y_con_func_integral[2*x_addition_datapoints:] -
                       y_con_func_integral[
                           :len(x_values_convolution) -
                           2*x_addition_datapoints] +
                       y_offset)
    return function_values

#####################################
# 3D nonlinear regression
#####################################


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

    function_names = ['quadratic_3D']
    assert function_type in function_names, 'Unknown function type.'

    # 'polynomial_3D': order of parameters: [0]*x^2 + [1]*y^2 + [2]*x*y + [3]*x
    # + [4]*y + [5]
    if function_type == function_names[0]:
        x_meshgrid, y_meshgrid = np.meshgrid(x_values, y_values)
        return (parameters[0]*x_meshgrid**2 + parameters[1]*y_meshgrid**2 +
                parameters[2]*x_meshgrid*y_meshgrid +
                parameters[3]*x_meshgrid + parameters[4]*y_meshgrid +
                parameters[5])
