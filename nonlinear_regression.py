# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy import integrate


#####################################
# 2D nonlinear regression
#####################################

def nonlinear_regression(x_values, y_values, function_type, z_values=None,
                         boundaries=None, initial_guess=None, max_iter=1000,
                         alg='evo'):
    assert alg in ['evo', 'lm'], ('No valid alg. Allowed values are \'%s\' '
                                  'and \'%s\'') % ('evo', 'lm')

    if alg == 'evo':
        return differential_evolution(
            fit_error, bounds=boundaries,
            args=(x_values, y_values, function_type, 'sum_of_squares'),
            maxiter=max_iter)
    if alg == 'lm':
        return least_squares(
            fit_error, initial_guess,
            args=(x_values, y_values, function_type, 'residuals'), method='lm')


def fit_error(fit_par, x_values, y_values, function_type,
              mode='sum_of_squares'):
    curr_values = calc_function(x_values, fit_par, function_type)

    modes = ['sum_of_squares', 'residuals']
    if mode == modes[0]:  # sum_of_squares
        return np.sum((curr_values - y_values)**2)
    elif mode == modes[1]:  # residuals
        return curr_values - y_values
    else:
        raise ValueError(
            'No valid fit error mode given, allowed values are {}.'.format(
                modes))


def calc_function(x_values, parameters, function_type):
    function_names = ['polynomial', 'Gauss', 'rectangle_gauss_convolution',
                      'Langmuir', 'triangle']

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
        function_values = np.sum(
            parameters[:, 0, np.newaxis] * np.exp(
                (x_values - parameters[:, 1, np.newaxis])**2 /
                (-2 * parameters[:, 3, np.newaxis]**2)) +
            parameters[:, 2, np.newaxis], axis=0)

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
        function_values = parameters[0]*x_values*parameters[1]/(
            1 + x_values*parameters[1])

    # 'triangle': Calculate a triangle function. The triangle function is zero
    # outside of the triangle and different slopes on both sides of the
    # triangle are possible.Order of parameters: The first number gives the x
    # value where the triangle starts. The second number gives the x value
    # where the triangle stops. The third number gives the x value for the
    # triangle maximum. The fourth value gives the y value of the triangle
    # maximum.
    elif function_type == function_names[4]:
        start_left = parameters[0]
        start_right = parameters[1]
        x_max = parameters[2]
        y_max = parameters[3]

        left_mask = np.logical_and(
            x_values >= min(start_left, x_max),
            x_values <= max(start_left, x_max))
        right_mask = np.logical_and(
            x_values > min(x_max, start_right),
            x_values <= max(x_max, start_right))

        left_slope = y_max/(x_max - start_left)
        right_slope = -y_max/(start_right - x_max)

        triangle = np.zeros_like(x_values)
        triangle[left_mask] = left_slope * (x_values[left_mask] - start_left)
        triangle[right_mask] = right_slope * (
            x_values[right_mask] - start_right)
        function_values = triangle

    else:
        raise ValueError('Unknown function type, allowed '
                         'values are {}'.format(function_names))

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
