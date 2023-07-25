# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:20:57 2022

@author: aso
"""

import numpy as np
import matplotlib.pyplot as plt

from pyDataFitting import principal_component_regression
from little_helpers.math_functions import gaussian

# The wavelength range of a spectrum
wavelengths = np.linspace(200, 400, 1000)

# The spectrum intensities are calculated as gaussian lines, the corresponding
# parameters are given here
amps = [[3, 2, 0.4], [2, 4, 3.3], [1.3, 4.2, 2.4]]
x_offs = [[220, 300, 350], [250, 280, 320], [265.3, 275, 285]]
y_offs = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
sigmas = [[2, 3, 1.1], [5, 1.5, 3.2], [2, 3.5, 7]]

# the intensities of the pure components are calculated
components = np.empty((len(amps), len(wavelengths)))
for idx, (curr_amp, curr_xo, curr_yo, curr_s) in enumerate(
        zip(amps, x_offs, y_offs, sigmas)):
    components[idx] = gaussian(
        wavelengths, curr_amp, curr_xo, curr_yo, curr_s)


# For the pricipal component regression, mixture sprectra of the different
# components will be calculated with the following factors.
mix_coeffs = np.array([[3, 2, 1.1],
                       [0.2, 0, 3.2],
                       [0.2, 0.8, 2.4],
                       [6, 2.1, 3.2],
                       [0.9, 0.2, 0.1]])

# The spectra of the mixtures are calculated.
mixtures = np.empty((len(mix_coeffs), len(wavelengths)))
for idx, curr_coeffs in enumerate(mix_coeffs):
    mixtures[idx] = (components.T*curr_coeffs).sum(axis=1)

# A principal component regression is performed
pcr = principal_component_regression(
    mixtures, y=mix_coeffs,
    x_names=np.round(wavelengths, 2).astype('str'),
    y_names=['Comp_{}'.format(idx) for idx in range(mix_coeffs.shape[1])])
pcr.pcr_fit(n_components=3)

# The mixing coefficients should be found again by using the predict method
prediction = pcr.predict(mixtures[[0]], 3)


fig1, ax1 = plt.subplots()
for idx, curr_comp in enumerate(components):
    ax1.plot(wavelengths, curr_comp, label='Component {}'.format(idx+1))
ax1.set_xlabel('Wavelength [nm]')
ax1.set_ylabel('Absorption [a.u.]')
ax1.legend()
ax1.set_title('Pure components')

fig2, ax2 = plt.subplots()
for idx, curr_mix in enumerate(mixtures):
    ax2.plot(wavelengths, curr_mix, label='Mixture{}'.format(idx+1))
ax2.set_xlabel('Wavelength [nm]')
ax2.set_ylabel('Absorption [a.u.]')
ax2.legend()
ax2.set_title('Mixture spectra')