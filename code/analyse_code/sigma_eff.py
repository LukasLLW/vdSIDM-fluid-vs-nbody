import numpy as np
from analyse_code.Moments import moments
from astropy import constants as ct
from astropy import units as ut

G_CONSTANT = ct.G


def V_max(r_s, rho_s):
    """Calculate the maximum velocity V_max in km/s.

    Parameters:
    r_s (astropy.units.Quantity): Scale radius, typically in kpc.
    rho_s (astropy.units.Quantity): Characteristic density, typically in
    M_sun/pc^3.

    Returns:
    float: Maximum velocity in km/s.
    """
    v_squared = G_CONSTANT * rho_s * (r_s**2)
    v = 1.649 * np.sqrt(v_squared)
    return v.to(ut.km / ut.s).value


def Velo_disp(r_s, rho_s):
    """Calculate the velocity dispersion in km/s.

    Parameters:
    r_s (astropy.units.Quantity): Scale radius, typically in kpc.
    rho_s (astropy.units.Quantity): Characteristic density, typically in
    M_sun/pc^3.

    Returns:
    float: Velocity dispersion in km/s.
    """
    return V_max(r_s, rho_s) *0.64


def sigma_eff(r_s, rho_s, w, sigma_0):
    """Calculate V_max, velocity dispersion, and the effective cross-section.

    Parameters:
    r_s (astropy.units.Quantity): Scale radius, typically in kpc.
    rho_s (astropy.units.Quantity): Characteristic density, typically in
    M_sun/pc^3.
    w (float): Scaling parameter for the moment argument in km/s.
    sigma_0 (astropy.units.Quantity): Cross-section normalization, typically in
    cm^2/g.

    Returns:
    dict: Dictionary containing V_max [km/s], Velo_disp [km/s], and sigma_eff
    [cm^2/g].
    """
    K_5 = lambda x: moments("Fischer_5")(x)

    v_max_val = V_max(r_s, rho_s)
    v_disp_val = Velo_disp(r_s, rho_s)
    argument = v_disp_val / w
    sigma_eff_val = sigma_0 * K_5(argument)

    return {
        "V_max [km/s]": v_max_val,
        "Velo_disp [km/s]": v_disp_val,
        "sigma_eff [cm^2/g]": sigma_eff_val.to(ut.cm**2 / ut.g).value,
        "K_5": K_5(argument),
    }