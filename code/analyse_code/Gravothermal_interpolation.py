import numpy as np
from scipy.interpolate import RegularGridInterpolator

def build_inverse_r_mapping(t_arr, r_arr, num_r_samples=200):
    """Maps (t, physical_r) -> normalized_r [0, 1]"""
    r_actual_min = np.max(r_arr[:, 0])  
    r_actual_max = np.min(r_arr[:, -1])   
    regular_r_actual = np.linspace(r_actual_min, r_actual_max, num_r_samples)  
    
    normed_r = np.linspace(0, 1, r_arr.shape[1])
    
   
    inverted_normed_r_grid = np.array([
        np.interp(regular_r_actual, r_arr[i, :], normed_r) for i in range(len(t_arr))
    ])

    return RegularGridInterpolator(
        (t_arr, regular_r_actual), 
        inverted_normed_r_grid, 
        bounds_error=False, 
        fill_value=None
    )

def build_value_mapping(t_arr, value_arr):
    """Maps (t, normalized_r) -> physical_value"""
    normed_r = np.linspace(0, 1, value_arr.shape[1])
    
    return RegularGridInterpolator(
        (t_arr, normed_r), 
        value_arr, 
        bounds_error=False, 
        fill_value=None
    )

def get_interpolation(t_arr1d, r_arr2d, value_arr2d):

    r_to_norm_func = build_inverse_r_mapping(t_arr1d, r_arr2d)
    
    norm_to_v_func = build_value_mapping(t_arr1d, value_arr2d)
    return lambda t, r: norm_to_v_func(np.column_stack((t, r_to_norm_func(np.column_stack((t, r))))))