[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# pyRegression
Linear and nonlinear fit functions that can be used *e.g.* for curve fitting.
Is not meant to duplicate methods already implemented *e.g.* in NumPy or SciPy,
but to provide additional, specialized regression methods or higher computation
speed. You will need certain functions of my little_helpers repository and
quite a few other, external packages like Numpy, Pandas, matplotlib,
scikit-learn, statsmodels, Scipy.

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
* polynomial_fit: Allows to perform polynomial fits by minimizing the sum of the squared residuals while also taking equality constaints into account via Lagrange multiplicators. This can be used to force the regression function through certain points or to force it to have certain slopes at a given points. Also does unconstrained polynomial fits, but is slower than the corresponding Numpy functions.
* piecewise_polynomial_fit: Allows to do a picewise polynomial fit on a dataset, *i.e.* the data is divided into segments that are then each fitted with an own polynomial function. The segments can be fitted with polynomials of different orders. It is possible to use equality constraints on the segment borders, so that the segments *e.g.* are forced to have the same y values at the borders or the same slopes.
* segment_regression: Does a piecewise polynomial fit with the segment borders, y values at the segment borders, or the slopes at the segment borders as additional fit parameters. The additional fit parameters are estimated with an evolutionary fitting algorithm which calls picewise_polynomial_fit several times in each iteration, so the whole procedure is rather slow (albeit still very usable).

## General nonlinear regression (in nonlinear_regression.py)
* nonlinear_regression: Does nonlinear regressions by minimizing the sum of the squared residuals. Basically utilizes differential_evolution from scipy.optimize to estimatze the parameters of complex regression functions. The functions must be included in calc_functions, but can be added easily there. This is not a particularly fast method, so use methods from other packages for simple problems.
* nonlinear_regression_3D: Does the same like nonlinear_regression, but on 3D datasets. Also here, the regression function must be included in calc_function_3D.

## Principal component regression and partial least squares regression (in multivariate_regression.py)
* principal_component_regression: A class for a principal component regression (PCR). Does a principal component analysis of the dataset and a multilinear regression on the resulting scores with one or several responses in order to generate a model to predict the responses from future data. The PCR parts work quite well, the methods included for generating various plots still need improving.
* pls_regression: A class to help with using the partial least squares regression class from scikit-learn. It is usable, but could do with some redesigning.
