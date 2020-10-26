# pyRegression
Linear and nonlinear fit functions that can be used *e.g.* for curve fitting.
Is not meant to duplicate methods already implemented *e.g.* in NumPy or SciPy,
but to provide additional, specialized regression methods or higher computation
speed.

## Linear regression (in linear_regression.py)
* dataset_regression: Does a classical linear least squares regression. Treats
the input data as a linear combination of the different components from
reference data. Can be used for example to fit spectra of mixtures with spectra
of pure components. Produces the same result like, but much faster than using
sklearn.linear_model.LinearRegression().fit(...).
* lin_reg_all_sections: Does linear regressions on a dataset starting with the
first two datapoints and expands the segment by one for each iteration. The
regression metrics are useful to determine if a dataset behaves linearly at its
beginning or not, and when a transition to nonlinear behavior occurs.

## Polynomial regression (in polynomial_regression.py)

## General nonlinear regression (in nonlinear_regression.py)

## Principal component regression and partial least squares regression (in multivariate_regression.py)

