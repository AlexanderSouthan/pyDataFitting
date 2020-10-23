# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy import integrate


#####################################
# 2D nonlinear regression
#####################################

def polynomial_fit(x_values, y_values, poly_order, fixed_points=None,
            fixed_slopes=None):
    """
    Fit a dataset with a polynomial function including constraints.
    
    The fit uses Lagrange multiplicators to introduce equality
    constraints after formulating the polynomial fit as a
    multilinear fit problem. The least squares of the residuals
    is minimized upon the fit. The fit polynomial is described by
    the following expression: 

        y_f = a0 + a1*x + a2*x^2 + a3*x^3 + ... + an*x^n
            = [1   x   x^2   x^3  ...  x^n]*[ a0 ]
                                            [ a1 ]
                                            [ a2 ]
                                            [ a3 ]
                                            [....]
                                            [ an ]

    The minimization problem of the unconstrained fit is:
        d(X*a-y)^2 = 0
    with X as a matrix containing as many rows as data points to be
    fitted, given by the vector y:
        X = [1   x   x^2   x^3  ...  x^n]
            [1   x   x^2   x^3  ...  x^n]
            [... ... ...   ...       ...]
            [1   x   x^2   x^3  ...  x^n]
    The set of linear equations to be solved for a is then given by:
        X(T)*X*a = X(T)*y
    with (T) meaning the transposed matrix.

    Constraints are introduced by:
        y_c = C*a
    with y_c as a vector of y values and C as the matrix to
    calculate y_c with a. The Lagrangian with Lagrange
    multiplicators lambda is constructed and its minimum is found
    by:
        [2*X(T)*X   C(T)] * [  a   ] = [2*X(T)*y]
        [  C          0 ]   [lambda]   [   b    ]
        
    The maximum number of constraints that can be introduced is
    given by poly_order+1.

    Parameters
    ----------
    x_values : ndarray
        The x values (independent variable) used for the fit. Must
        be a 1D array.
    y_values : ndarray
        The y values (dependent variable) used for the fit. Must be
        a 1D array with the same length as x_values.
    poly_order : int
        The polynomial order used for the fit.
    fixed_points : list of tuples or None, optional
        Contains constraints for points that the fit functions must
        pass through. Each point is given by a tuple of two numbers,
        the x and the y corrdinate of the point. If no point
        constraints are to be applied, this must be None. The
        default is None.
    fixed_slopes : list of tuples or None, optional
        Contains constraints for slopes that the fit functions must
        have at specific x values. Each slope is given by a tuple of
        two numbers, the x value and the slope. If no slope
        constraints are to be applied, this must be None. The
        default is None.
    Returns
    -------
    y_fit : ndarray
        A 1D array containing the y values at the x values in
        x_values.
    coefs : ndarray
        The vector a from the linear equations given above. Can be
        passed directly to np.polynomial.polynomial.polyval to
        calculate the polynomial values.

    """
    # Sort x and y values given in tuples from fixed_points into
    # individual arrays. The numbers are converted to float64 explicitly to
    # avoid problems with higher order polynomials where the numbers quickly
    # are too big e.g. for int32 which is automatically used for integer
    # constraints.
    if fixed_points is not None:
        x_points = np.array(
            [curr_point[0] for curr_point in fixed_points]).astype('float64')
        y_points = np.array(
            [curr_point[1] for curr_point in fixed_points]).astype('float64')
    else:
        x_points = np.array([])
        y_points = np.array([])

    # Sort x and slope values given in tuples from fixed_slopes into
    # individual arrays
    if fixed_slopes is not None:
        x_slopes = np.array(
            [curr_slope[0] for curr_slope in fixed_slopes])
        slopes = np.array(
            [curr_slope[1] for curr_slope in fixed_slopes])
    else:
        x_slopes = np.array([])
        slopes = np.array([])

    # The number of constraints and the unknown coefficients in the
    # polynomial fit function
    constraint_number = len(x_slopes) + len(x_points)
    coef_number = poly_order + 1

    # A matrix with len(x_values) rows and poly_order columns
    # containing the x_values to the power of all exponents included
    # by poly_order
    x_matrix = x_values[:, np.newaxis]**np.arange(coef_number)

    # The upper left part of the matrix in the equation system to be
    # solved later on
    ul_matrix = 2*np.dot(x_matrix.T, x_matrix)

    # A matrix containing the factors in front of the polynomial
    # coefficients for the constraints...
    # ...in case of point constraints the fixed x values to the
    # power of all exponents included in poly_order
    constraints_matrix = x_points[:, np.newaxis]**np.arange(coef_number)
    # ...in case of slope constraints the x_slopes multiplied with
    # the exponents and with the x_values to power of the
    # exponents-1.
    constraints_matrix = np.append(
        constraints_matrix,
        np.arange(coef_number)*x_slopes[:, np.newaxis]**
        np.insert(np.arange(coef_number-1), 0, 0),
        axis=0)

    # The combined matrix used for calculation of the least squares
    # solution
    combined_matrix = np.zeros(
        (len(ul_matrix)+len(constraints_matrix),
         len(ul_matrix)+len(constraints_matrix)))
    combined_matrix[:coef_number, :coef_number] = ul_matrix
    combined_matrix[:coef_number:, coef_number:] = constraints_matrix.T
    combined_matrix[coef_number:, :coef_number] = constraints_matrix

    # The upper part of the vector used for calculating the least
    # squares solution.
    upper_vector = 2 * np.dot(x_matrix.T, y_values)

    # The combined vector used for calulating the least squares
    # solution.
    combined_vector = np.concatenate(
        [upper_vector, y_points, slopes])

    # Solution of the set of linear equations leading to the
    # polynomial coefficients with minimized least squares of the
    # residuals, possibly with constraints.
    coefs = np.linalg.solve(
        combined_matrix, combined_vector)[:coef_number]
    y_fit = np.polynomial.polynomial.polyval(x_values, coefs)
    return (y_fit, coefs, x_matrix, ul_matrix, constraints_matrix, combined_matrix, upper_vector, combined_vector)


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
              mode='sum_of_squares', par_type='list'):
    par_types = ['list', 'dict']
    if par_type == par_types[0]:  # list
        curr_values = calc_function(x_values, fit_par, function_type)
    elif par_type == par_types[1]:
        curr_values = calc_function_with_dict(x_values, fit_par, function_type)
    else:
        raise ValueError('No valid value for par_type defining the data '
                         'structure of the parameters given. Allowed values '
                         'must be in {}.'.format(par_types))

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
