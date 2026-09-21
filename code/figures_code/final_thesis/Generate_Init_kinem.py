import matplotlib.pyplot as plt
from data_path import figure_Path
import numpy as np
from rich.console import Console
from analyse_code.Moments import moments
from figures_code.final_thesis.generell_dics import (PALETTES, STYLES, create_legend, get_kwargs)
from figures_code.final_thesis.LOAD_GRAVOTHERMAL_DATA import (Gravothermal_data_by_keys,)
from figures_code.final_thesis.LOAD_NBODY_DATA import Nbody_data_by_keys
import matplotlib.transforms as mtransforms

console = Console()
plt.style.use(r".\figures_code\style_format_one.mplstyle")

cfg_Gravothermal = Gravothermal_data_by_keys(("A", "B"), ("dual", "eff"))
cfg_Nbody = Nbody_data_by_keys(("A", "B"))

fig, ax = plt.subplots(2, 1, sharex=True)
K_5 = lambda x: moments("Fischer_5")(x)
ax_dispersion = ax[0]
ax_avrg_cross = ax[1]

for cfg in cfg_Gravothermal:
    kwargs = get_kwargs(cfg)
    ini_rho = cfg["rho"][0,:]
    ini_p = cfg["p"][0,:]
    r = cfg["r"][0,:]

    dispersion = np.sqrt(ini_p/ini_rho)
    ax_dispersion.loglog(r, dispersion, **kwargs)

    if cfg["model"] == "dual":
        ax_avrg_cross.loglog(r, cfg["sigma_0"]*K_5(dispersion/20), **kwargs)
    elif cfg["model"] =="eff":
        ax_avrg_cross.hlines(cfg["sigma_eff [cm^2/g]"],0,1e2, **kwargs)

for cfg in cfg_Nbody:
    kwargs = get_kwargs(cfg)
    dispersion = cfg["v"][0,:]
    err = np.clip(cfg["v_err"][0,:],0,1e9)
    r = cfg["r"]
    
    ax_dispersion.plot(r, dispersion, **kwargs)
    ax_dispersion.fill_between(
        r,
        dispersion - err,
        dispersion + err,
        alpha=STYLES["nbody_benchmark"]["alpha_fill"],
        edgecolors="black",
        color=kwargs["color"],
    )

ax_dispersion.set_ylabel(r"$\nu(r) \quad [\text{km/s}]$")
ax_avrg_cross.set_ylabel(r"$\langle \sigma_\text{v}\rangle \quad \Big [\text{cm}^2\text{/g} \Big]$")
ax_avrg_cross.set_xlabel(r"$r \quad [\text{kpc}]$")
ax_avrg_cross.set_ylim(1e1,10**(3.7))

create_legend(
    ax_avrg_cross, 
    cfg_Nbody, 
    cfg_Gravothermal, 
    extra=[[], []], 
    ADD_RUN=True, 
    space=0.08,
    loc="upper right",
    bbox=(1, -0.4)
)

plt.tight_layout()
fig.subplots_adjust(bottom=0.25)

fig.canvas.draw()
bbox = fig.get_tightbbox(fig.canvas.get_renderer())
bbox_inches = bbox.padded(0)


standard_x_pad = 0.1  
bbox_inches.x0 -= standard_x_pad
bbox_inches.x1 += standard_x_pad

bbox_inches.y0 -= 0.5
bbox_inches.y1 += 0.5

ax_avrg_cross.set_xlim(0.008, 1e1)
ax_dispersion.set_xlim(1e-3, 1e1)
plt.savefig(
    figure_Path("Sec4_3_1-init_kinematics.png"),
    bbox_inches=bbox_inches,
)

console.print(f"[bold green]SUCCESS[/bold green] saved under: {figure_Path('Sec4_3_1-init_kinematics.png')}\n")
plt.show()