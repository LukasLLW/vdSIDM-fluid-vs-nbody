import matplotlib.pyplot as plt
from data_path import figure_Path
import numpy as np
from rich.console import Console
from analyse_code.Moments import moments
from figures_code.final_thesis.generell_dics import (PALETTES,STYLES,create_legend,get_kwargs)
from figures_code.final_thesis.LOAD_GRAVOTHERMAL_DATA import (Gravothermal_data_by_keys,)
from figures_code.final_thesis.LOAD_NBODY_DATA import Nbody_data_by_keys

console = Console()
plt.style.use(r".\figures_code\style_format_one.mplstyle")

cfg_Gravothermal = Gravothermal_data_by_keys(("A", "B"), ("dual", "eff"))
cfg_Nbody = Nbody_data_by_keys(("", ""))

fig, ax_kn = plt.subplots(1, 1, sharex=True)

K_5 = lambda x: moments("Fischer_5")(x)

G_CONSTANT = 6.67e-11  # m^3 kg^-1 s^-2
ax_kn.set_xlim(1e-3, 1e1)
for cfg in cfg_Gravothermal:
    
    kwargs = get_kwargs(cfg)
    
    ini_rho = cfg["rho"][0,:]
    ini_p = cfg["p"][0,:]
    r = cfg["r"][0,:]

    dispersion = np.sqrt(ini_p/ini_rho)
    rho_conversion = 1.988e+30 / (3.0857e+19)**3

    if cfg["model"] == "dual":
        sigma= cfg["sigma_0"]*K_5(dispersion/20)
    elif cfg["model"] =="eff":
        sigma=cfg["sigma_eff [cm^2/g]"]
    y= np.sqrt(4 * np.pi * G_CONSTANT * ini_rho * rho_conversion / (dispersion * 1e3)**2) * \
            1.0 / (ini_rho * rho_conversion * sigma * 1e-1)
    ax_kn.loglog(r, y, **kwargs)


ax_kn.set_ylabel(r"Kn$(r)$")
ax_kn.set_xlabel(r"$r \quad [\text{kpc}]$")


x_min, x_max = ax_kn.get_xlim()
y_min, y_max = ax_kn.get_ylim()

ax_kn.fill_between([0,1e4], 1, 1e15, color="gray", alpha=0.15, zorder=0)
ax_kn.fill_between([0,1e4], 1e-15, 1, color="lightgray", alpha=0.1, zorder=0)

ax_kn.text(0.02, 0.95, r"$\text{Kn} \gg 1$", transform=ax_kn.transAxes, fontsize=10, verticalalignment='top')
ax_kn.text(0.02, 0.05, r"$\text{Kn} < 1$", transform=ax_kn.transAxes, fontsize=10, verticalalignment='top')

ax_kn.set_xlim(x_min, x_max)
ax_kn.set_ylim(y_min, y_max)

create_legend(
    ax_kn, cfg_Nbody, cfg_Gravothermal, extra=[[], []],   ADD_RUN=True, space=0.05
)
plt.savefig(figure_Path("Sec4_3_1-init_kn.png"))
console.print(f"[bold green]SUCCESS[/bold green] saved under: {figure_Path("Sec4_3_1-init_kn.png")}\n")



plt.show()