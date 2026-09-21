import matplotlib.pyplot as plt

plot_styles = {
    "nbody_benchmark": {
        "linestyle": "-",
        "linewidth": 1.5,
        "alpha_fill": 0.15, 
    },
    "standard_effective": {
        "linestyle": "--",
        "marker": "o",
        "markersize": 1,
        "zorder":3,
        "c":1
    },
    "dual_regime": {
        "linestyle": ":",
        "marker": "o",
        "markerfacecolor": "none",  
        "markeredgewidth": 1.5,
        "markersize": 1,
        "zorder":4,
        "c":0
    },
    "numerical_pipeline": {
        "linestyle": "-",
        "linewidth": 2,
        "marker": "None",
         "zorder":-1,
         "c":2
    },
    "profile_time_rescaling": {
        "linestyle": "-.",
        "marker": "d",
        "markersize": 1,
         "zorder":2,
         "c":3
    },
    "alternative_core": {
        "linestyle": "None",
        "marker": "^",
        "markersize": 7,
         "zorder":7
    }
}


import colorsys
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def get_4d_parameter_color(
    moment: float,
    sigma: float,
    alpha: float=1,
    c_val: float=1,
    max_blend: float = 0.85,
) -> tuple:

    norm_moment = (max(3.0, min(7.0, moment)) - 3.0) / 4.0
    norm_sigma = (max(3000.0, min(8000.0, sigma)) - 3000.0) / 8000.0

    alpha_norm = max(0.5, min(1.5, alpha))
    c_norm = max(0.5, min(1.5, c_val))

    try:
        base_cmap = plt.colormaps["plasma"]
    except AttributeError:
        base_cmap = plt.cm.get_cmap("plasma")

    base_rgb = base_cmap(norm_moment)[:3]

    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    adjusted_sigma = norm_sigma**alpha_norm

    l_final = l + (max_blend - l) * adjusted_sigma
    s_final = s * (1.0 - 0.4 * adjusted_sigma) * c_norm
    s_final = max(0.0, min(1.0, s_final))

    r_final, g_final, b_final = colorsys.hls_to_rgb(h, l_final, s_final)

    return (r_final, g_final, b_final, 1.0)


