import numpy as np
import scipy.ndimage as ndimage

def avrg_density(values, error_arr, r_arr):
    """Calculates weighted density in log10 space to prevent numerical overflow."""
    if len(values) == 0:
        return np.nan, np.nan, np.nan

    safe_values = np.where(values <= 0, 1e-10, values)
    safe_error = np.where(error_arr <= 0, 1e-10, error_arr)

    log_rho_individual = np.log10(safe_values)

    log_error_individual = safe_error / (safe_values * np.log(10.0))
    log_error_individual = np.clip(log_error_individual,0,0.05)
    weights = 1.0 / (log_error_individual**2)
    sum_weights = np.sum(weights)

    if sum_weights == 0:
        return np.nan, np.nan, np.mean(r_arr)

    log_rho_avg = np.sum(weights * log_rho_individual) / sum_weights
    log_error_avg = 1.0 / np.sqrt(sum_weights)

    rho = 10**log_rho_avg

    error = log_error_avg * rho * np.log(10.0)
    error = np.clip(error, a_min=0, a_max=1e10)

    r_calculated = np.mean(r_arr)

    return rho, error, r_calculated

def analyze_and_fit_data(
    time, rho, sigma, range_min, range_thresh, target_base_val
):
    """Analyzes N-body data and extracts robust milestone points mapped back to raw data."""
    time = np.array(time)
    rho = np.array(rho)
    rho_smoothed = ndimage.gaussian_filter1d(rho, sigma=sigma)

    idx_min_range = (time >= range_min[0]) & (time <= range_min[1])
    poly_min = np.polyfit(time[idx_min_range], rho[idx_min_range], 2)
    t_fit_min = np.linspace(range_min[0], range_min[1], 500)
    rho_fit_min = np.polyval(poly_min, t_fit_min)

    t_poly_min = t_fit_min[np.argmin(rho_fit_min)]
    idx_raw_min = np.argmin(np.abs(time - t_poly_min))

    t_exact_min = time[idx_raw_min]
    rho_exact_min = rho[idx_raw_min]

    idx_thresh_range = (time >= range_thresh[0]) & (time <= range_thresh[1])
    poly_thresh = np.polyfit(
        time[idx_thresh_range], rho_smoothed[idx_thresh_range], 2
    )
    t_fit_thresh = np.linspace(range_thresh[0], range_thresh[1], 500)
    rho_fit_thresh = np.polyval(poly_thresh, t_fit_thresh)

    critical_value = 1.05 * target_base_val
    thresh_fit_idx = np.argmin(np.abs(rho_fit_thresh - critical_value))

    t_poly_thresh = t_fit_thresh[thresh_fit_idx]
    idx_raw_thresh = np.argmin(np.abs(time - t_poly_thresh))

    t_exact_thresh = time[idx_raw_thresh]
    rho_exact_thresh = rho[idx_raw_thresh]

    local_rho_mean = np.mean(rho[idx_thresh_range])
    if np.abs(rho_exact_thresh - local_rho_mean) / local_rho_mean > 0.5:
        print(
            f"Warning: Fitted threshold rho ({rho_exact_thresh:.2e}) deviates significantly from local raw data."
        )

    return t_exact_min, rho_exact_min, t_exact_thresh, rho_exact_thresh