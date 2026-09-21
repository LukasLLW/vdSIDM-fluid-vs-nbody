import numpy as np
from scipy import integrate, interpolate
from collections.abc import Callable

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
G = 1.0 

DEBUG = True

def PHI_GRAVITATIONAL(r_arr: np.ndarray, rho_arr: np.ndarray):
    """
    Computes the relative potential Psi with boundary Psi(infinity) = 0.

    The density profile is assumed to be known up to r_max.
    Outside r_max, the enclosed mass is treated as constant,
    so the potential follows a Kepler tail Psi = G M_tot / r.
    """

    rho_func = interpolate.CubicSpline(r_arr, rho_arr, extrapolate=None)

    integrand_1 = lambda r_prime: r_prime**2 * rho_func(r_prime)
    integrand_2 = lambda r_prime: r_prime * rho_func(r_prime)

    PSI_at_r = []
    escape_velo = []

    r_max = r_arr[-1]

    for r in r_arr:
        if r == 0:
            part_1 = 0.0
        else:
            part_1 = (1.0 / r) * integrate.quad(integrand_1, 0.0, r)[0]

        part_2 = integrate.quad(integrand_2, r, r_max)[0]

        psi_val = G * (part_1 + part_2)

        PSI_at_r.append(psi_val)
        escape_velo.append(np.sqrt(2.0 * max(psi_val, 0.0)))

    PSI_at_r = np.array(PSI_at_r)
    escape_velo = np.array(escape_velo)

    PSI_r_spline = interpolate.CubicSpline(r_arr, PSI_at_r, extrapolate=None)

    psi_for_rho = PSI_at_r.copy()
    rho_for_rho = rho_arr.copy()

    sort_idx = np.argsort(psi_for_rho)
    psi_sorted = psi_for_rho[sort_idx]
    rho_sorted = rho_for_rho[sort_idx]

    psi_sorted = np.concatenate(([0.0], psi_sorted))
    rho_sorted = np.concatenate(([0.0], rho_sorted))

    PSI_rho_spline = interpolate.PchipInterpolator(
        psi_sorted,
        rho_sorted,
        extrapolate=None
    )

    return PSI_r_spline, PSI_rho_spline, escape_velo


def EPSILON(v, PHI: Callable):
    '''
    Computes the specific energy for a given velocity and gravitational potential.
    following the eq 4.41 of Galactic Dynamics (Binney & Tremaine, 2nd ed.).
    '''
    return PHI - 0.5 * v**2


def Escape_velocity(r_arr : np.ndarray, rho_arr : np.ndarray):
    '''
    Computes the escape velocity for a given radius and density profile.
    following the eq 2.31 of Galactic Dynamics (Binney & Tremaine, 2nd ed.).
    '''    
    return np.sqrt(2 * abs(PHI_GRAVITATIONAL(r_arr, rho_arr)[0](r_arr)))


def f_Eddington(r_arr: np.ndarray, rho_arr: np.ndarray, n: int = 100):
    '''
    Computes the Eddington distribution function for a given density profile.
    following the eq 4.46b of Galactic Dynamics (Binney & Tremaine, 2nd ed.).

    NOTE: nu = rho
    '''
    print("1")
    PSI_r, PSI_rho, escape_velo = PHI_GRAVITATIONAL(r_arr, rho_arr)
    psi = PSI_r(r_arr)

    tol = 1e-12
    print("2")
    if np.any(psi < -tol):
        print("min psi =", psi.min())
        raise ValueError("PSI contains significantly negative values.")

    psi = np.maximum(psi, 0.0)

    d_rho_d_PSI = PSI_rho.derivative()
    d2_rho_d_PSI2 = d_rho_d_PSI.derivative()

    if DEBUG:
        fig, ax = plt.subplots(1, 1)
        import seaborn as sns
        color_palette= sns.color_palette("mako", n_colors=3)
        ax.plot(PSI_r(r_arr), PSI_rho(PSI_r(r_arr)), label=rf"$\rho(\psi)$", color=color_palette[0], zorder=1)
        ax.plot(PSI_r(r_arr), d_rho_d_PSI(PSI_r(r_arr)), label=rf"$\text{{d}}\rho/\text{{d}}\psi(\psi_r(r))$", color=color_palette[1], zorder=1)
        ax.loglog(PSI_r(r_arr), d2_rho_d_PSI2(PSI_r(r_arr)), label=rf"$\text{{d}}^2\rho/\text{{d}}\psi^2(\psi_r(r))$", color=color_palette[2], zorder=1)        

        ax.set_xlabel(r"$\psi$")
        ax.set_ylabel(r"$\rho /\rho_0$")

        ax_r = ax.twinx()  
        ax_r.plot(psi, r_arr, label=r"$r(\psi)$", color="tab:red", linestyle="--", zorder=-3)
        ax_r.set_ylabel(r"$r/r_s$")
        ax_r.set_yscale("log")
        ax_r.set_ylim(9.5e1,1e-3)
        lines_ax, labels_ax = ax.get_legend_handles_labels()
        lines_ax_r, labels_ax_r = ax_r.get_legend_handles_labels()
        ax.legend(lines_ax + lines_ax_r, labels_ax + labels_ax_r, loc="best")

        from data_path import figure_Path
        name = "Sec4_3_5-Eddington_psi_curve.png"
        plt.savefig(figure_Path(name), bbox_inches="tight")  
        plt.show()

    nr = len(r_arr)
    v_escape = escape_velo
    v_frac = np.linspace(0.0, 1.0, n)


    integrand = lambda PSI, EPSILON: d2_rho_d_PSI2(PSI) / np.sqrt(EPSILON - PSI)

    part_2_func = lambda EPSILON: (
        d_rho_d_PSI(0.0) / np.sqrt(EPSILON)
        if EPSILON > 0
        else 0.0
    )

    edd_grid = []
    EPSILON_GRID = []
    f_prefactor = 1.0 / (8.0 * np.sqrt(2.0) * np.pi**3)
    velocity_grid = []
    
    for i in range(nr):
        place_holder = []
        psi_i = max(PSI_r(r_arr[i]), 0.0)
        v_esc_i = np.sqrt(2.0 * psi_i)

        velo_arr = np.linspace(0.0, v_esc_i * (1.0), 100)
        velocity_grid.append(velo_arr)
        ep_place = []

        for v in velo_arr:
            current_epsilon = EPSILON(v, psi_i)
            ep_place.append(current_epsilon)
            if current_epsilon > 1e-12:
                PART_1 = integrate.quad(
                    integrand,
                    0.0,
                    current_epsilon,
                    args=(current_epsilon,),
                    epsabs=1e-5
                )[0]

                PART_2 = part_2_func(current_epsilon)

                place_holder.append( f_prefactor * (PART_1 + PART_2))
            else:
                place_holder.append( 0.0 )
        edd_grid.append(place_holder)
        EPSILON_GRID.append(ep_place)

    edd_grid = np.array(edd_grid)
    edd_grid = np.where(np.isfinite(edd_grid), edd_grid, 0.0)
    edd_grid = np.maximum(edd_grid, 0.0)

    EPSILON_GRID = np.array(EPSILON_GRID)
    velocity_grid = np.array(velocity_grid)

    speed_distribution = 4.0 * np.pi * velocity_grid**2 * edd_grid

    unnormed = 4.0 * np.pi * velocity_grid**2 * edd_grid
    integrals = np.array([integrate.trapezoid(unnormed[i], velocity_grid[i]) for i in range(nr)])[:, None]
    probability_grid = np.divide(unnormed, integrals, out=np.zeros_like(unnormed), where=integrals > 0)
    return edd_grid, EPSILON_GRID, velocity_grid, speed_distribution, probability_grid, escape_velo


def f_Maxwell_Boltzmann(r_arr: np.ndarray, rho_arr: np.ndarray, velocity_grid, PSI_r):
    """
    Computes the local Maxwell-Boltzmann velocity distribution across the entire grid.
    The local velocity dispersion sigma^2(r) is derived via the Jeans Equation.
    Returns an equivalent probability_grid for easy 2D plotting.
    """
    nr = len(r_arr)
    probability_grid_mb = np.zeros_like(velocity_grid)
    
    r_quad_integrand = r_arr**2 * rho_arr
    m_arr = np.array([
        integrate.simpson(r_quad_integrand[:i+1], x=r_arr[:i+1]) if i > 0 else 0.0
        for i in range(nr)
    ])
    
    dpsi_dr_arr = np.zeros_like(r_arr)
    dpsi_dr_arr[1:] = m_arr[1:] / (r_arr[1:]**2)
    dpsi_dr_arr[0] = dpsi_dr_arr[1] 
    
    jeans_integrand = rho_arr * dpsi_dr_arr
    
    for i in range(nr):
        rho_r = rho_arr[i]
        if rho_r <= 1e-10 or i >= nr - 2:
            continue
            
        integral_val = integrate.simpson(jeans_integrand[i:], x=r_arr[i:])
        sigma2 = (1.0 / rho_r) * integral_val
        
        if sigma2 <= 1e-10:
            continue
            
        v_arr = velocity_grid[i]
        sigma = np.sqrt(sigma2)
        prefactor = np.sqrt(2 / np.pi) * (1.0 / sigma**3)
        
        p_v_raw = prefactor * (v_arr**2) * np.exp(-v_arr**2 / (2 * sigma2))
        
        norm_mb = integrate.trapezoid(p_v_raw, v_arr)
        if norm_mb > 0:
            probability_grid_mb[i, :] = p_v_raw / norm_mb

    return probability_grid_mb

def compute_local_G_average(
    r_idx: int, 
    PSI_r, 
    r_arr: np.ndarray, 
    f_grid: np.ndarray, 
    velocity_grid: np.ndarray, 
    G_func: Callable
) -> float:
    """
    Calculates the local expectation value <G(v_rel)> at a fixed radius index 
    for an arbitrary phase-space density f(E) via 3D integration.
    
    Parameters:
    -----------
    r_idx : int
        Index of the desired radius in r_arr.
    PSI_r : CubicSpline
        Spline of the gravitational potential Psi(r).
    r_arr : np.ndarray
        Array of normalized radii.
    f_grid : np.ndarray
        2D array of the phase-space density f(E) (e.g., edd_grid, mb_grid).
    velocity_grid : np.ndarray
        2D array of velocities.
    G_func : Callable
        The function G(v_rel) whose expectation value is to be computed.
    """
    psi_local = PSI_r(r_arr[r_idx])
    v_grid_local = velocity_grid[r_idx]
    f_local = f_grid[r_idx]
    
    f_interp = interpolate.CubicSpline(v_grid_local, f_local, extrapolate=False)
    
    def integrand(mu, v2, v1):
        f1 = f_interp(v1)
        f2 = f_interp(v2)
        
        if np.isnan(f1) or np.isnan(f2) or f1 <= 0 or f2 <= 0:
            return 0.0
            
        v_rel = np.sqrt(v1**2 + v2**2 - 2.0 * v1 * v2 * mu)
        if v_rel <= 0:
            return 0.0
            
        return (v1**2) * (v2**2) * f1 * f2 * G_func(v_rel)

    v_max = v_grid_local[-1]
    
    options = {'epsabs': 1e-4, 'epsrel': 1e-4}
    
    integral_val, _ = integrate.tplquad(
        integrand,
        0.0, v_max,                        
        lambda v1: 0.0, lambda v1: v_max,  
        lambda v1, v2: -1.0, lambda v1, v2: 1.0,  
        **options
    )
    
    rho_local = integrate.trapezoid(4.0 * np.pi * v_grid_local**2 * f_local, v_grid_local)
    
    if rho_local <= 1e-10:
        return 0.0
        
    total_prefactor = 8.0 * np.pi**2
    
    return (total_prefactor * integral_val) / (rho_local**2)

def compute_local_G_average_fast(
    r_idx: int, 
    f_grid: np.ndarray, 
    velocity_grid: np.ndarray, 
    G_func: Callable,
    n_angle: int = 50
) -> float:
    """
    Highly optimized and vectorized local expectation value <G(v_rel)>.
    Collapses the 3D integral into a fast 2D grid integration.
    All comments in English as requested.
    """
    v_local = velocity_grid[r_idx]
    f_local = f_grid[r_idx]
    
    rho_local = integrate.trapezoid(4.0 * np.pi * v_local**2 * f_local, v_local)
    if rho_local <= 1e-10:
        return 0.0

    V1, V2 = np.meshgrid(v_local, v_local, indexing='ij')
    F1, F2 = np.meshgrid(f_local, f_local, indexing='ij')

    mu_arr = np.linspace(-1.0, 1.0, n_angle)
    
    G_mu_values = []

    for mu in mu_arr:
        v_rel = np.sqrt(V1**2 + V2**2 - 2.0 * V1 * V2 * mu)

        valid = v_rel > 0
        G_val = np.zeros_like(v_rel)
        G_val[valid] = G_func(v_rel[valid])

        G_mu_values.append(G_val)

    G_mu_values = np.array(G_mu_values)
    G_integrated = integrate.trapezoid(G_mu_values, x=mu_arr, axis=0)

    integrand_2d = (V1**2) * (V2**2) * F1 * F2 * G_integrated

    int_v2 = integrate.trapezoid(integrand_2d, v_local, axis=1)
    integral_val = integrate.trapezoid(int_v2, v_local, axis=0)

    total_prefactor = 8.0 * np.pi**2
    
    return (total_prefactor * integral_val) / (rho_local**2)

def visc_sigma(x):
    """
    Calculates the viscous cross-section without the normalization constant sigma_c.
    Input: x = v_rel / w
    """
    return (1.0 + x**2)**(-2)

def compute_Kp(
    r_idx: int,
    PSI_r,
    r_arr: np.ndarray,
    f_grid: np.ndarray,
    velocity_grid: np.ndarray,
    p: float,
    w: float
) -> float:
    """
    Calculates the normalized transfer coefficient K_p at a fixed radius index.
    
    K_p = < sigma_visc(v_rel/w) * v_rel^p > / < v_rel^p >
    
    Parameters:
    -----------
    r_idx : int
        Index of the desired radius in r_arr.
    PSI_r : CubicSpline
        Spline of the gravitational potential Psi(r).
    r_arr : np.ndarray
        Array of normalized radii.
    f_grid : np.ndarray
        2D array of the phase-space density f(E) (e.g., edd_grid, mb_grid).
    velocity_grid : np.ndarray
        2D array of velocities.
    p : float
        Power index of the relative velocity.
    w : float
        Characteristic scale velocity of the cross-section.
    """
    G_numerator = lambda v_rel: visc_sigma(v_rel / w) * (v_rel**p)
    
    G_denominator = lambda v_rel: v_rel**p
    
    print(f"Calculating numerator expectation value for K_p at radius index {r_idx}...")
    numerator_avg = compute_local_G_average(
        r_idx, PSI_r, r_arr, f_grid, velocity_grid, G_numerator
    )
    
    print(f"Calculating denominator expectation value for K_p at radius index {r_idx}...")
    denominator_avg = compute_local_G_average(
        r_idx, PSI_r, r_arr, f_grid, velocity_grid, G_denominator
    )
    
    if denominator_avg <= 1e-10:
        return 0.0
        
    return numerator_avg / denominator_avg

def compute_Kp_fast(
    r_idx: int,
    f_grid: np.ndarray,
    velocity_grid: np.ndarray,
    p: float,
    w: float,
    n_angle: int = 10
) -> float:
    """
    Fast K_p calculation utilizing the vectorized grid integration.
    """
    print("kp")
    G_numerator = lambda v_rel: visc_sigma(v_rel / w) * (v_rel**p)
    G_denominator = lambda v_rel: v_rel**p
    
    numerator_avg = compute_local_G_average_fast(
        r_idx, f_grid, velocity_grid, G_numerator, n_angle=n_angle
    )

    denominator_avg = compute_local_G_average_fast(
        r_idx, f_grid, velocity_grid, G_denominator, n_angle=n_angle
    )
    if denominator_avg <= 1e-10:
        return 0.0
        
    return numerator_avg / denominator_avg

def perform_lbm_bgk_step(
    f_grid_old: np.ndarray,
    f_eq_grid: np.ndarray,
    velocity_grid: np.ndarray,
    rho_profile: np.ndarray,
    sigma_func: Callable,
    sigma_0: float,
    w,
    dt: float,
    n_angle: int = 50
) -> np.ndarray:
    """
    Executes the local Lattice Boltzmann BGK relaxation step for all radii.
    Computes the interaction rate 1/tau(r) internally and updates the 
    phase-space distribution function grid.
    All comments in English for thesis consistency.

    Parameters:
    -----------
    f_grid_old : np.ndarray
        2D array (n_radius x n_velocity) of the phase-space density from the previous step.
    f_eq_grid : np.ndarray
        2D array (n_radius x n_velocity) of the target equilibrium distribution (Maxwell-Boltzmann).
    velocity_grid : np.ndarray
        2D array (n_radius x n_velocity) of the velocity coordinates.
    rho_profile : np.ndarray
        1D array (n_radius) of the current local matter density rho(r).
    sigma_func : Callable
        The velocity-dependent cross-section function sigma(x), where x = v_rel / w.
    sigma_0 : float
        The constant cross-section normalization factor.
    dt : float
        The simulation time step size Delta t.
    n_angle : int, optional
        Number of angular grid points for the mu integration. Default is 50.

    Returns:
    --------
    f_grid_new : np.ndarray
        2D array (n_radius x n_velocity) of the updated phase-space density.
    """
    n_radius, n_velocity = f_grid_old.shape
    f_grid_new = np.copy(f_grid_old)
    
    mu_arr = np.linspace(-1.0, 1.0, n_angle)
    mu_weight = 2.0 / (n_angle - 1)

    for r_idx in range(n_radius):
        v_local = velocity_grid[r_idx]
        f_local = f_grid_old[r_idx]

        rho_local = integrate.trapezoid(
            4.0 * np.pi * v_local**2 * f_local,
            v_local
        )

        if rho_local <= 1e-10:
            continue

        V1, V2 = np.meshgrid(v_local, v_local, indexing='ij')
        F1, F2 = np.meshgrid(f_local, f_local, indexing='ij')
        kernel_integrated = np.zeros_like(V1)

        for mu in mu_arr:
            v_rel = np.sqrt(V1**2 + V2**2 - 2.0 * V1 * V2 * mu)
            valid = v_rel > 0
            kernel_slice = np.zeros_like(v_rel)
            kernel_slice[valid] = sigma_0 * sigma_func(v_rel[valid] / w) * v_rel[valid]
            kernel_integrated += kernel_slice

        kernel_integrated *= mu_weight

        integrand_2d = (V1**2) * (V2**2) * F1 * F2 * kernel_integrated
        int_v2 = integrate.simpson(integrand_2d, x=v_local, axis=1)
        total_integral = integrate.simpson(int_v2, x=v_local, axis=0)

        total_prefactor = 4.0 * np.pi**2
        one_over_tau = (total_prefactor * total_integral) / rho_local

        alpha = 1.0 - np.exp(-dt * one_over_tau)

        f_grid_new[r_idx] = (
            (1.0 - alpha) * f_grid_old[r_idx]
            + alpha * f_eq_grid[r_idx]
        )
        f_grid_new[r_idx] = np.where(
            np.isfinite(f_grid_new[r_idx]),
            f_grid_new[r_idx],
            0.0
        )

        f_grid_new[r_idx] = np.maximum(f_grid_new[r_idx], 0.0)
    return f_grid_new


def build_mb_bgk_grid(
    r_grid,
    rho_profile,
    disp_profile,
    n_v=500,
    n_sigma=12.0
):
    r_grid = np.asarray(r_grid, dtype=float)
    rho_profile = np.asarray(rho_profile, dtype=float)
    disp_profile = np.asarray(disp_profile, dtype=float)

    nr = len(r_grid)

    velocity_grid_mb = np.zeros((nr, n_v))
    probability_grid_mb = np.zeros((nr, n_v))
    f_grid_mb = np.zeros((nr, n_v))

    for i in range(nr):
        rho_i = rho_profile[i]
        sig_i = disp_profile[i]

        if (
            not np.isfinite(rho_i)
            or not np.isfinite(sig_i)
            or rho_i <= 0.0
            or sig_i <= 0.0
        ):
            continue

        v_max = n_sigma * sig_i
        v = np.linspace(0.0, v_max, n_v)

        P = (
            np.sqrt(2.0 / np.pi)
            * v**2
            / sig_i**3
            * np.exp(-v**2 / (2.0 * sig_i**2))
        )

        norm = integrate.trapezoid(P, v)

        if not np.isfinite(norm) or norm <= 0.0:
            continue

        P = P / norm

        f = np.zeros_like(v)
        mask = v > 0.0

        f[mask] = (
            rho_i
            * P[mask]
            / (4.0 * np.pi * v[mask]**2)
        )

        velocity_grid_mb[i, :] = v
        probability_grid_mb[i, :] = P
        f_grid_mb[i, :] = f

    return {
        "velocity_grid_mb": velocity_grid_mb,
        "probability_grid_mb": probability_grid_mb,
        "f_grid_mb": f_grid_mb
    }

def build_mb_on_existing_grid(rho_profile, disp_profile, velocity_grid):
    rho_profile = np.asarray(rho_profile, dtype=float)
    disp_profile = np.asarray(disp_profile, dtype=float)

    f_grid_mb = np.zeros_like(velocity_grid)

    for i in range(len(rho_profile)):
        rho_i = rho_profile[i]
        sig_i = disp_profile[i]
        v = velocity_grid[i]

        if (
            not np.isfinite(rho_i)
            or not np.isfinite(sig_i)
            or rho_i <= 0.0
            or sig_i <= 0.0
        ):
            continue

        sigma2 = sig_i**2

        f_raw = (
            1.0
            / ((2.0 * np.pi * sigma2)**1.5)
            * np.exp(-v**2 / (2.0 * sigma2))
        )

        norm = integrate.trapezoid(
            4.0 * np.pi * v**2 * f_raw,
            v
        )

        if not np.isfinite(norm) or norm <= 0.0:
            continue

        f_grid_mb[i] = rho_i * f_raw / norm

    return f_grid_mb