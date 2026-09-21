import matplotlib.pyplot as plt
from data_path import figure_Path
import numpy as np
from rich.console import Console
from analyse_code.Moments import moments
from figures_code.final_thesis.generell_dics import (PALETTES, STYLES, create_legend, get_kwargs)
from figures_code.final_thesis.LOAD_GRAVOTHERMAL_DATA import (Gravothermal_data_by_keys, base)
from figures_code.final_thesis.LOAD_NBODY_DATA import Nbody_data_by_keys

console = Console()
plt.style.use(r".\figures_code\style_format_one.mplstyle")

cfg_Gravothermal = Gravothermal_data_by_keys(("B"), ("dual", ""))
cfg_Nbody = Nbody_data_by_keys(( "B"))
third_cfg = base(r"C:\Users\lukas\Documents\GitHub_new26\Bachelor-Code\Evaluation\Data_Archive\Halo1_sig6593d89_V2", key="B")
fith_cfg = cfg_Gravothermal[0]
seventh_cfg = base(r"C:\Users\lukas\Documents\GitHub_new26\Bachelor-Code\Evaluation\Data\run_k7", key="B")

all_cvg = [third_cfg, fith_cfg, seventh_cfg]

fig, ax = plt.subplots()

C = PALETTES["Transition 2"](3)
moment_labels = ["3rd moment", "5th moment", "7th moment"]



for i, cfg in enumerate(cfg_Nbody):
    kwargs = get_kwargs(cfg)
    x = cfg["t"]
    y = cfg["rho_avrg"]

    ax.semilogy(x, y, **kwargs)
    ax.fill_between(
        x,
        y - cfg["rho_avrg_err"],
        y + cfg["rho_avrg_err"],
        alpha=STYLES["nbody_benchmark"]["alpha_fill"],
        edgecolors="black",
        color=kwargs["color"],
    )
    x_max = max(x)
    y_max = max(y)



for i, cfg in enumerate(all_cvg):
    kwargs = get_kwargs(cfg)
    kwargs["color"] = C[i]

    x_data = cfg["t"]
    y_data = cfg["f_rho"](x_data, np.full_like(x_data, 0.0775))
    
    ax.semilogy(x_data, y_data, **kwargs)

    ax.text(
        x_data[-1] * 1.01 if x_data[-1]<x_max else x_max*0.88, 
        y_data[-1] * 0.6 if y_data[-1]<y_max else y_max*0.86, 
        moment_labels[i], 
        color=C[i], 
        va="center", 
        ha="left",
        fontsize=9
    )


create_legend(
    ax, cfg_Nbody, cfg_Gravothermal, [[], []]
)

ax.set_xlabel(r"$t \quad[\text{Gyr}]$")
ax.set_ylabel(r"$\rho \quad \Big [M_\odot/\text{kpc}^3 \Big]$")
ax.set_xlim(None, x_max)
ax.set_ylim(None, y_max)
name="Sec4_3_4-3_5_7_moment_t_r_profile.png"
plt.savefig(figure_Path(name))
console.print(f"[bold green]SUCCESS[/bold green] saved under: {figure_Path(name)}\n")

if __name__ == "__main__":
    plt.show()
else:
    plt.close()