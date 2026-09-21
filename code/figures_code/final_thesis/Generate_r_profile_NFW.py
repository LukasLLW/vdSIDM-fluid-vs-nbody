import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from data_path import figure_Path
import numpy as np
from rich.console import Console

from figures_code.final_thesis.generell_dics import (
    PALETTES,
    STYLES,
    create_legend,
    get_kwargs,
)
from figures_code.final_thesis.LOAD_GRAVOTHERMAL_DATA import (
    Gravothermal_data_by_keys,
)
from figures_code.final_thesis.LOAD_NBODY_DATA import Nbody_data_by_keys
from matplotlib.lines import Line2D
import os
import sys
import shutil
import matplotlib as mpl


console = Console()

plt.style.use(r'.\figures_code\style_format_one.mplstyle')

cfg_Gravothermal = Gravothermal_data_by_keys(("A"), ("dual", ""))
cfg_Nbody = Nbody_data_by_keys(("C","A", "B"))
r_goal = 0.0775

fig, ax = plt.subplots(1, 1, sharey=True)

ax_ins = ax.inset_axes([0.62, 0.6, 0.35, 0.35])

for i, cfg in enumerate(cfg_Gravothermal):
    kwargs = get_kwargs(cfg)
    x = cfg["r"][0]
    y = cfg["rho"][0,:]
    ax.loglog(x,y, **kwargs)
    ax_ins.loglog(x,y, **kwargs)

for i, cfg in enumerate(cfg_Nbody):
    kwargs = get_kwargs(cfg)
    x = cfg["r"]
    y = cfg["rho_2d"][0,:]

    mask = y > 1e4
    x =x[mask]
    y = y[mask]
    err = np.clip(cfg["rho_err_2d"][0,:],0,1e9)
    err=err[mask]
    ax.loglog(x,y, **kwargs)
    ax_ins.loglog(x,y, **kwargs) 

    for axis in [ax, ax_ins]:
        axis.fill_between(
            x,
            y - err,
            y + err,
            alpha=STYLES["nbody_benchmark"]["alpha_fill"],
            edgecolors="black",
            color=kwargs["color"],
        )

r_analytic = np.logspace(-6, 2, 100)
nfw_profile = lambda x: 44175e3 / ((x / 1.28) * (1 + x / 1.28)**2)

line, =ax.plot(r_analytic, nfw_profile(r_analytic), linestyle="-", color="black", alpha=0.7, linewidth=2, zorder=-1)
ax_ins.plot(r_analytic, nfw_profile(r_analytic), linestyle="-", color="black", alpha=0.7, linewidth=2, zorder=-1)

ax.set_xlim(10**(-3.2),1e1)
ax.set_ylim(nfw_profile(1e1),1e11)


x1, x2 = 1.2e-3, 10**(-1.4)
y1, y2 = 1e9, nfw_profile(1e-3)-10**(0.2)
ax_ins.set_xlim(x1, x2)
ax_ins.set_ylim(y1, y2)

ax_ins.set_xscale('log')
ax_ins.set_yscale('log')

ax_ins.tick_params(axis='both', which='both', labelsize=8) 

ax.indicate_inset_zoom(ax_ins, edgecolor="black", alpha=0.3)


ax.set_xlabel(r"$r \quad \left [\text{\textcolor{red}{kpc}} \right ]$")
ax.set_ylabel(r"$\rho \quad \Big[ \mathrm{M}_\odot / \mathrm{kpc}^3 \Big]$")

create_legend(
    ax, cfg_Nbody, cfg_Gravothermal, extra=[[line], ["Analytic NFW"]], bbox=(0.0,0.02), loc="lower left", ADD_RUN=True
)
plt.savefig(figure_Path("Sec4_1_1-radius_profiles_NFW.png"))
console.print(f"[bold green]SUCCESS[/bold green] saved under: {figure_Path("Sec4_1_1-radius_profiles_NFW.png")}\n")


print(nfw_profile(0.0775))
plt.show()