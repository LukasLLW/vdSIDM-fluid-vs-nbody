import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from data_path import figure_Path
import numpy as np
from rich.console import Console

from figures_code.final_thesis.generell_dics import (PALETTES, STYLES, create_legend, get_kwargs)
from figures_code.final_thesis.LOAD_GRAVOTHERMAL_DATA import (Gravothermal_data_by_keys,)
from figures_code.final_thesis.LOAD_NBODY_DATA import Nbody_data_by_keys
from data_path import gravothermal_root
from matplotlib.lines import Line2D
from matplotlib.colors import SymLogNorm 

console = Console()
plt.style.use(r".\figures_code\style_format_one.mplstyle")

cfg_Nbody = Nbody_data_by_keys(("A", "B"))

fig, axes = plt.subplots(1, 2, sharey=True)

C = PALETTES["Diff"]

all_betas = [cfg["beta"] for cfg in cfg_Nbody]
max_abs_val = max(np.max(np.abs(b)) for b in all_betas)


linthresh = 0.01 

custom_norm = SymLogNorm(
    linthresh=linthresh, 
    vmin=-max_abs_val, 
    vmax=max_abs_val, 
    base=10
)
import matplotlib.patheffects as pe
text_with_outline = [pe.withStroke(linewidth=1, foreground="black")]

#for ax in axes:
    #ax.set_xscale("log")
for i, cfg in enumerate(cfg_Nbody):
    t = cfg["t"]       
    r = cfg["r"]       
    beta = cfg["beta"] 

    T_grid, R_grid = np.meshgrid(t, r)
    ax = axes[i]
    ax.set_xlim(0, max(t))
    ax.set_ylim(min(r), max(r))
    ax.vlines(cfg["tcmin"][0], min(r), max(r), color="white")
    ax.vlines(cfg["tcc"][0], min(r), max(r), color="white")
    bbox_props = dict(boxstyle="round,pad=0.3", fc="none", ec="white", lw=1)

    ax.text(
        cfg["tcmin"][0]*1.05, 0.95, r"$t_\text{cf}$", 
        color="white", transform=ax.get_xaxis_transform(),
        va="top", ha="left",
        path_effects=text_with_outline
    )
    
    ax.text(
        cfg["tcc"][0]*.85, 0.95, r"$t_\text{cc}$", 
        color="white", transform=ax.get_xaxis_transform(),
        va="top", ha="left",
        path_effects=text_with_outline
    )
    
    mesh = ax.pcolormesh(T_grid, R_grid, beta.T, cmap=C, norm=custom_norm, shading='auto')
    
    ax.set_xlabel(r"$t \quad [\text{Gyr}]$ ")

axes[0].set_title("Simulation A")
axes[1].set_title("Simulation B")

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label(r"$\beta$")
#axes[0].set_xlabel(r"$ r \quad [\text{kpc}]$")
axes[0].set_ylabel(r"$ r \quad [\text{kpc}]$")
name="Sec4_3_5-beta.png"
fig.savefig(figure_Path(name))
console.print(f"[bold green]SUCCESS[/bold green] saved under: {figure_Path(name)}\n")


if __name__ == "__main__":
    plt.show()
else:
    plt.close()