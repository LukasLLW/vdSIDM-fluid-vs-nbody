from analyse_code.Moments import moments
import numpy as np
from astropy import units as ut

def new_t(sigma_0, sigma_const, nu, time):
    K_5 = lambda x: moments("Fischer_5")(x)

    nu_dim = nu* (ut.kpc / ut.Gyr)
    v_rms_core = nu_dim.to(ut.km / ut.s).value
    dt_sim=np.diff(time, prepend=0)
    dt = dt_sim*sigma_const/(sigma_0 * K_5(np.array(v_rms_core)/20))

    return np.cumsum(dt)