# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:17:41 2019

@author: AlMaMi
"""

import numpy as np
import pandas as pd
from numbers import Number
from scipy.stats import linregress

def lin_reg_all_sections(x_values,y_values,mode = 'all_values',r_squared_limit = None):
    """Calculates linear regressions for all sections of the input x and y values. 
    
    Starts always with first value and expands the segment by one for each iteration. 
    
    Returns either a DataFrame with slopes, intercepts and r_squared values for all segments (mode = 'all_values'), or slope and intercept at limit given for r_squared (mode = 'values_at_limit'), or both (mode = 'both')."""
    
    assert len(x_values) == len(y_values),'x and y must have same lengths, but have %i and %i' % (len(x_values),len(y_values))
    assert mode in ['all_values','values_at_limit','both'],'mode must either be \'%s\', \'%s\', or \'s\'' % ('all_values','values_at_limit','both')
    if mode in ['values_at_limit','both']:
        assert isinstance(r_squared_limit,Number),'invalid value for r_squared_limit'
    
    slopes = np.empty_like(x_values,dtype='float64')
    intercepts = np.empty_like(x_values,dtype='float64')
    r_squared = np.empty_like(x_values,dtype='float64')
    
    for ii in np.arange(1,len(y_values)):
        curr_slope, curr_intercept, curr_r_value, curr_p_value, curr_std_err = linregress(x_values[:ii+1],y_values[:ii+1])
        slopes[ii] = curr_slope
        intercepts[ii] = curr_intercept
        r_squared[ii] = curr_r_value**2
    
    results_df = pd.DataFrame(np.array([slopes,intercepts,r_squared]).T,columns = ['slopes','intercepts','r_squared'])
    
    if mode in ['values_at_limit','both']:
        mask = r_squared > r_squared_limit
        x_at_limit = x_values[mask][-1]
        y_at_limit = y_values[mask][-1]
        slope_at_limit = slopes[mask][-1]
        intercept_at_limit = intercepts[mask][-1]
        
    if mode == 'all_values':
        return_value = results_df
    elif mode == 'values_at_limit':
        return_value = (x_at_limit,y_at_limit,slope_at_limit,intercept_at_limit)
    elif mode == 'both':
        return_value = (results_df,x_at_limit,y_at_limit,slope_at_limit,intercept_at_limit)
        
    return return_value