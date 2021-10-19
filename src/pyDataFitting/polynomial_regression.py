# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import differential_evolution

from little_helpers.array_tools import segment_xy_values


def segment_regression(x_values, y_values, poly_orders, border_bounds,
                       y_bounds=None, slope_bounds=None, max_iter=1000):
    """
    Do a regression with a piecewise polynomial fit on the segmentation points.

    This method is useful if the segmentation points for a piecewise polynomial
    fit are not exactly known, but can be guessed to some extent. Generally, it
    is possible to give the whole data range as boundaries, so effectively no
    boundary conditions are passed to the fit, however this is not recommended
    and may result in an error message. It is better to provide separated
    boundaries for the segmentation points.

    Parameters
    ----------
    x_values : ndarray
        A 1D array with the length M holding the independent varibale used for
        the fit.
    y_values : ndarray
        A 1D array with the length M holding the dependent varibale used for
        the fit.
    poly_orders : list of int
        A list containing the polynomial orders used for the fit. Must contain
        one more element than border_bounds.
    border_bounds : list of tuples
        The range the fit varies the x values of the segmentation points. Each
        tuple has two entries and defines the range for one segmentation point,
        so the list has as many tuples as segmentation points are present. The
        first value in the tuple is the lower boundary, the second the upper
        boundary for the respective segmentation point.
    y_bounds : None, or list of tuples or None, optional
        The range the fit varies the y values of the segmentation points. Each
        tuple has two entries and defines the range for one segmentation point,
        so the list has as many tuples or None as segmentation points are
        present. The first value in the tuple is the lower boundary, the second
        the upper boundary for the respective segmentation point. Default is
        None which means no point constraint at the segmentation point.
    slope_bounds : None, or list of tuples or None, optional
        The range the fit varies the slope values at the segmentation points.
        Each tuple has two entries and defines the range for one segmentation
        point, so the list has as many tuples or None as segmentation points
        are present. The first value in the tuple is the lower boundary, the
        second the upper boundary for the respective segmentation point.
        Default is None which means no point constraint at the segmentation
        point.
    max_iter : int, optional
        The maximum numer of iterations performed in the iterative fit
        algorithm (currently differential evolution). The default is 1000.

    Returns
    -------
    y_fit : ndarray
        A 1D array containing the y values of the fit at the x values given in
        x_values.
    coefs : list of ndarray
        A list containing the coefficient vectors of the polynomial equations
        for the data segments. Each list entry can be passed directly to
        np.polynomial.polynomial.polyval to calculate the polynomial values.
    evo_fit : scipy.optimize.OptimizeResult
        The result of the differential evolution fit.

    """
    # border_bounds, y_bounds and slope_bounds are mixed together in boundaries
    # because the parameters for differential_evolution have to be one 1D
    # array. constraint_types passes a key to sum_of_squared_residuals to
    # identify which parameter is which kind of constraint.
    boundaries = []
    constraint_types = []
    for idx, curr_border_bounds in enumerate(border_bounds):
        boundaries.append(curr_border_bounds)
        constraint_types.append('b')
        if (y_bounds is not None) and (y_bounds[idx] is not None):
            boundaries.append(y_bounds[idx])
            constraint_types.append('y')
        if (slope_bounds is not None) and (slope_bounds[idx] is not None):
            boundaries.append(slope_bounds[idx])
            constraint_types.append('s')

    # The evolutionary fit itself.
    evo_fit = differential_evolution(
        sum_of_squared_residuals, bounds=boundaries,
        args=(x_values, y_values, poly_orders, constraint_types),
        maxiter=max_iter)

    # segment_borders and y_at_borders are extracted from the 1D evo_fit
    # array so that they can be used in the calculation of the fit curve.
    segment_borders, y_at_borders, slope_at_borders = decode_fit_par(
        evo_fit.x, constraint_types)

    y_fit, coefs = piecewise_polynomial_fit(
        x_values, y_values, segment_borders, poly_orders,
        y_at_borders=y_at_borders, slope_at_borders=slope_at_borders)

    return (y_fit, coefs, evo_fit)


def sum_of_squared_residuals(fit_par, x_values, y_values, poly_orders,
                             constraint_types):
    """
    Objective function to be minimized during segment_regression.

    The function is called by differential_evolution.

    Parameters
    ----------
    fit_par : ndarray
        A 1D array containing the fit parameters. Contains segment_borders,
        and possibly y_at_borders and slope at borders directly following the
        corresponding segment border.
    x_values : ndarray
        A 1D array with the length M holding the independent varibale used for
        the fit.
    y_values : ndarray
        A 1D array with the length M holding the dependent varibale used for
        the fit.
    poly_orders : list of int
        A list containing the polynomial orders used for the fit. Must contain
        one more element than segment_borers contained in fit_par.
    constraint_types : list of str
        Encodes which value in fit_par is which kind of constraint. 'b' means
        segmentation border, 'y' means a common y value of adjacent segments,
        and 's' means a common slope of adjacent segments.

    Returns
    -------
    sum of squared residuals
        The value to be minimized during the fit.

    """
    segment_borders, y_at_borders, slope_at_borders = decode_fit_par(
        fit_par, constraint_types)

    curr_values, curr_coefs = piecewise_polynomial_fit(
        x_values, y_values, segment_borders, poly_orders,
        y_at_borders=y_at_borders, slope_at_borders=slope_at_borders)

    return np.sum((curr_values - y_values)**2)


def decode_fit_par(fit_par, constraint_types):
    """
    Translate the 1D array of fit parameters to individual lists.

    The fit paramters are sorted into the lists segment_borders, y_at_borders
    slope_at_borders based on the strings given in constraint_types. The three
    lists are used as arguments in piecewise_polynomial_fit called in the
    functions sum_of_squared_residuals and segment_regression.

    Parameters
    ----------
    fit_par : ndarray
        A 1D array containing the fit parameters. Contains segment_borders,
        and possibly y_at_borders and slope at borders directly following the
        corresponding segment border.
    constraint_types : list of str
        Encodes which value in fit_par is which kind of constraint. 'b' means
        segmentation border, 'y' means a common y value of adjacent segments,
        and 's' means a common slope of adjacent segments.

    Returns
    -------
    segment_borders, y_at_borders, slope_at_borders
        The arguments used for piecewise_polynomial_fit, see its docstring for
        details.

    """
    # Fit parameters from segment_regression and sum_of_squared_residuals are
    # reordered to be understood by piecewise_polynomial_fit.
    segment_borders = []
    y_at_borders = []
    slope_at_borders = []
    for idx, curr_type in enumerate(constraint_types):
        if curr_type == 'b':
            segment_borders.append(fit_par[idx])
            y_at_borders.append(None)
            slope_at_borders.append(None)
        if curr_type == 'y':
            y_at_borders[-1] = fit_par[idx]
        if curr_type == 's':
            slope_at_borders[-1] = fit_par[idx]

    return (segment_borders, y_at_borders, slope_at_borders)


def piecewise_polynomial_fit(x_values, y_values, segment_borders,
                             poly_orders, y_at_borders=None,
                             slope_at_borders=None):
    """
    Piecewise polynomial fit using polynomial_fit for the data segments.

    Since polynomial_fit allows constraints such as fixed points or fixed
    slopes, this can be used for the piecewise polynomial fit as well. By
    default, the data is divided into segments at the values given in
    segment_borders and the segments are fitted each by a polynomial fit using
    the least squares method. Thus, discontinuities will arise at the segment
    borders between the fits of the segments. These discontinuities can be
    suppressed by applying equality constraints to the fits that force the fit
    curves through specific points or to have specific slopes. The method
    allows to use polynomials of different orders for the different segments.

    Parameters
    ----------
    x_values : ndarray
        A 1D array with the length M holding the independent varibale used for
        the fit.
    y_values : ndarray
        A 1D array with the length M holding the dependent varibale used for
        the fit.
    segment_borders : list of int or float
        The values with respect to x_values at which the data is divided into
        segments. An arbitrary number of segment borders may be given, but it
        is recommended to provide a sorted list in order to avoid confusion.
        If the list is not sorted, it will be sorted and the sorting is also
        applied to y_at_borders and slope_at_borders (if they are given), but
        not to poly_orders.
    poly_orders : list of int
        A list containing the polynomial orders used for the fit. Must contain
        one more element than segment_borders.
    y_at_borders : None, or list of float or None, optional
        May contain dependent variable values used as equality constraints at
        the segment borders. The fits of both touching segments are forced
        through the point given by the pair (segment border, y_at_border). The
        list entries may also be None to state that at a certain segment
        border, no constraint is to be applied. The default is None which means
        that no contraints are applied for any segment border.
    slope_at_borders : None, or list of float or None, optional
        Analogous to y_at_borders, with the difference that here, fixed slopes
        at the segment borders are given. The default is None.

    Returns
    -------
    y_fit : ndarray
        A 1D array containing the y values of the fit at the x values given in
        x_values.
    coefs : list of ndarray
        A list containing the coefficient vectors of the polynomial equations
        for the data segments. Each list entry can be passed directly to
        np.polynomial.polynomial.polyval to calculate the polynomial values.

    """

    sort_order = np.argsort(segment_borders)
    segment_borders = np.array(segment_borders)[sort_order]

    # Fixed points are given by the x values in segment_borders and the y
    # values given in y_at_borders and are collected in tuples of the two
    # numbers or in empty tuples if no fixed point is used for the given
    # segment border.
    if y_at_borders is not None:
        y_at_borders = np.array(y_at_borders)[sort_order]
        fixed_points = [()]  # for left edge
        for x, y in zip(segment_borders, y_at_borders):
            curr_point = (x, y) if y is not None else ()
            fixed_points.append(curr_point)
        fixed_points.append(())  # for right edge
    else:
        fixed_points = [()] * (len(segment_borders) + 2)

    # Fixed slopes are given by the x values in segment_borders and the slope
    # values given in slope_at_borders and are collected in tuples of the two
    # numbers or in empty tuples if no fixed slope is used for the given
    # segment border.
    if slope_at_borders is not None:
        fixed_slopes = [()]  # for left edge
        for x, slope in zip(segment_borders, slope_at_borders):
            curr_slope = (x, slope) if slope is not None else ()
            fixed_slopes.append(curr_slope)
        fixed_slopes.append(())  # for right edge
    else:
        fixed_slopes = [()] * (len(segment_borders) + 2)

    x_segments, y_segments = segment_xy_values(x_values, segment_borders,
                                               y_values=y_values)

    fit_segments = []
    coefs = []
    # In the loop, the segments are fitted individually, one in each iteration.
    for curr_x, curr_y, curr_order, left_fix, right_fix, left_slope, right_slope in zip(
            x_segments, y_segments, poly_orders, fixed_points[:-1],
            fixed_points[1:], fixed_slopes[:-1], fixed_slopes[1:]):
        # The fixed points and slopes are translated into the syntax understood
        # by polynomial_fit.
        curr_fixed = []
        if left_fix:
            curr_fixed.append(left_fix)
        if right_fix:
            curr_fixed.append(right_fix)
        if not curr_fixed:
            curr_fixed = None
        curr_slope = []
        if left_slope:
            curr_slope.append(left_slope)
        if right_slope:
            curr_slope.append(right_slope)
        if not curr_slope:
            curr_slope = None

        # The polynomial fit itself.
        curr_segment, curr_coefs = polynomial_fit(
            curr_x, curr_y, curr_order, fixed_points=curr_fixed,
            fixed_slopes=curr_slope)

        # Fit results of the segments are collected in two lists.
        fit_segments.append(
            curr_segment if len(fit_segments) == len(x_segments)-1
            else curr_segment[:-1])
        coefs.append(curr_coefs)

    # The fit curves of the segments are stitched together.
    y_fit = np.concatenate(fit_segments)

    return (y_fit, coefs)


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

    There is currently a small problem that occurred during testing with a
    random input for x_values. The result was not numerically stable, slightly
    different fit curves were obtained with different random points.

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
    # constraint_number = len(x_slopes) + len(x_points)
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
        np.arange(coef_number) * x_slopes[:, np.newaxis] **
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

    return (y_fit, coefs)
