#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:03:24 2022

@author: Alexander Southan
"""

import numpy as np
from scipy.stats import percentileofscore
from scipy.special import erfinv


def percentiles(data):
    percentiles = np.empty_like(data)

    for ii, curr_data in enumerate(data):
        percentiles[ii] = percentileofscore(data, curr_data, kind='mean')

    return percentiles


def theo_residual_percentiles(residuals):
    # Calculation with the probit function,
    # see https://en.wikipedia.org/wiki/Probit
    theo_resid_percentiles = np.sqrt(2)*erfinv(2*percentiles(residuals)/100-1)

    return theo_resid_percentiles
