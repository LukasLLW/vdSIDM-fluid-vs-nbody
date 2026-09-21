import matplotlib.pyplot as plt
from data_path import figure_Path
import numpy as np
from rich.console import Console
from analyse_code.Moments import moments
from figures_code.final_thesis.generell_dics import (PALETTES,STYLES,create_legend,get_kwargs)
from figures_code.final_thesis.LOAD_GRAVOTHERMAL_DATA import (Gravothermal_data_by_keys,)
from figures_code.final_thesis.LOAD_NBODY_DATA import Nbody_data_by_keys
from analyse_code.Moments import moments
import matplotlib.colors as mcolors

console = Console()
plt.style.use(r".\figures_code\style_format_one.mplstyle")

K_5 = lambda x: moments("Fischer_5")(x)

cfg_Gravothermal = Gravothermal_data_by_keys(("B"), ("dual", "eff"))
cfg_Nbody = Nbody_data_by_keys((""))

fig, ax = plt.subplots()

ax.set_xscale("log")
ax.set_xlabel(r"$r \quad [\text{kpc}]$")
ax.set_yscale("log")
ax.set_ylabel(r"$\langle \sigma_v \rangle \quad \Big [\text{cm}^2 / \text{g} \Big]$")

cfg =cfg_Gravothermal[0]
idx = [0, int(cfg["tcf_idx"]/2),cfg["tcf_idx"], int(abs(cfg["tcf_idx"] - cfg["tcc_idx"])/2+cfg["tcf_idx"]), cfg["tcc_idx"]]
print(idx)
time_C = PALETTES["Transition 1"](5)[::-1]

for n,i in enumerate(idx):

    kwargs = get_kwargs(cfg)
    kwargs["color"] = time_C[n]
    ini_rho = cfg["rho"][i,:]
    ini_p = cfg["p"][i,:]
    r = cfg["r"][i,:]

    dispersion = np.sqrt(ini_p/ini_rho)
    ax.loglog(r, cfg["sigma_0"]*K_5(dispersion/20), **kwargs)

ax.hlines(62.2, 0, 1e1, linestyle= STYLES["eff"]["linestyle"], color="gray")
ax.set_xlim(1e-3, 1e1)
ax.set_ylim(3e1, 5e3)


create_legend(
   ax, cfg_Nbody, cfg_Gravothermal, [[], []], bbox=(0.98,0.98), loc="upper right"
)


cmap = mcolors.ListedColormap(time_C[::-1])
bounds = np.arange(5 + 1) + 0.5
norm = mcolors.BoundaryNorm(bounds, cmap.N)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, ticks=np.arange(5) + 1, pad=0.02)
cbar.set_label(r"$\leftarrow$ Evolution Time Step")

tick_labels = [r"$t_\text{cc}$", "", r"$t_\text{cf}$", "", "Init."]
cbar.ax.set_yticklabels(tick_labels)

ax.text(
        1.5e-3, 70 , 
        fr"$\sigma_{{\text{{eff}}}} =62.2 \text{{ cm}}^2/\text{{g}}$", 
        color="gray", 
        va="bottom", 
        ha="left"
    )

name="Sec4_3_3-Avrg_sigma_B.png"
plt.savefig(figure_Path(name))
console.print(f"[bold green]SUCCESS[/bold green] saved under: {figure_Path(name)}\n")


plt.show()