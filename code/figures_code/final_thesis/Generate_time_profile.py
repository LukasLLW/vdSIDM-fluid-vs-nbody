import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from data_path import figure_Path
import numpy as np
from rich.console import Console
from pathlib import Path
from figures_code.final_thesis.generell_dics import (PALETTES,STYLES,create_legend,get_kwargs)
from figures_code.final_thesis.LOAD_GRAVOTHERMAL_DATA import (Gravothermal_data_by_keys,)
from figures_code.final_thesis.LOAD_NBODY_DATA import Nbody_data_by_keys
from matplotlib.lines import Line2D

console = Console()
plt.style.use(Path(".")/"figures_code"/"style_format_one.mplstyle")

cfg_Gravothermal = Gravothermal_data_by_keys(("A", "B"), ("dual", "num","eff", "rescaling"))
cfg_Nbody = Nbody_data_by_keys(("A", "B"))
r_goal = 0.0775

fig, ax = plt.subplots(1, 2, sharey=True)
axes = {"A": ax[0], "B": ax[1]}

for i, cfg in enumerate(cfg_Gravothermal):
    kwargs = get_kwargs(cfg)
    x = cfg["t"][5:] if cfg["model"] in ("eff", "rescaling") else cfg["t"]
    y = cfg["f_rho"](x, np.full_like(x, r_goal)).flatten()  

    if cfg["run"] in ("A") and cfg["model"] in ("dual"):
        x_extra = np.linspace(x[-1], x[-1] * 1.1, 50)
        
        poly = np.polyfit(x[-30:], np.log(y[-30:]), deg=2)

        y_extra = np.exp(np.polyval(poly, x_extra))
        
        x = np.concatenate([x, x_extra[1:]])
        y = np.concatenate([y, y_extra[1:]])
        
    axes[cfg["run"]].semilogy(x, y, **kwargs)
    axes[cfg["run"]].scatter(
        cfg["tcmin"][0],
        cfg["tcmin"][1],
        marker="^",
        edgecolors="black",
        color=kwargs["color"],
        zorder=kwargs["zorder"]+1
    )
    axes[cfg["run"]].scatter(
        cfg["tcc"][0],
        cfg["tcc"][1],
        marker="D",
        edgecolors="black",
        color=kwargs["color"],
        zorder=kwargs["zorder"]+1
    )

for i, cfg in enumerate(cfg_Nbody):
    kwargs = get_kwargs(cfg)
    x = cfg["t"]
    y = cfg["rho_avrg"]

    axes[cfg["run"]].semilogy(x, y, **kwargs)
    axes[cfg["run"]].fill_between(
        x,
        y - cfg["rho_avrg_err"],
        y + cfg["rho_avrg_err"],
        alpha=STYLES["nbody_benchmark"]["alpha_fill"],
        edgecolors="black",
        color=kwargs["color"],
    )
    axes[cfg["run"]].scatter(
        cfg["tcmin"][0],
        cfg["tcmin"][1],
        marker="^",
        edgecolors="black",
        color=kwargs["color"],
        zorder=kwargs["zorder"]+1
    )
    axes[cfg["run"]].scatter(
        cfg["tcc"][0],
        cfg["tcc"][1],
        marker="D",
        edgecolors="black",
        color=kwargs["color"],
        zorder=kwargs["zorder"]+1
    )

Y_MIN = 10 ** (7.9)
Y_MAX = 10 ** (9.4)
axes["A"].set_ylim(Y_MIN, Y_MAX)
axes["B"].set_ylim(Y_MIN, Y_MAX)

axes["A"].set_ylabel(r"$\rho \quad \Big [M_\odot/\text{kpc}^3 \Big ]$")

x_max_0 = axes["A"].get_xlim()[1]
x_max_1 = axes["B"].get_xlim()[1]
width_ratios = [x_max_0, x_max_1]

axes["A"].text(0.1, 10 ** (9.3), "RUN A", color=PALETTES["Analytic"][0])
axes["B"].text(0.1, 10 ** (9.3), "RUN B", color=PALETTES["Analytic"][0])

gs = axes["A"].get_gridspec()
gs.set_width_ratios(width_ratios)
fig.tight_layout()

invisible_handle = mpatches.Rectangle(
    (0, 0), 0, 0, fill=False, edgecolor="none", visible=False
)
marker_handles = [
    Line2D(
        [0],
        [0],
        marker="^",
        color="none",
        markerfacecolor="gray",
        markeredgecolor="black",
        markersize=8,
        linestyle="none",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        color="none",
        markerfacecolor="gray",
        markeredgecolor="black",
        markersize=8,
        linestyle="none",
    ),
    invisible_handle,
]
marker_labels = [r"$t_{\text{cf}}$", r"$t_{\text{cc}}$", ""]

create_legend(
    axes["A"], cfg_Nbody, cfg_Gravothermal, [marker_handles, marker_labels]
)
for e in ax:
    e.set_xlabel(rf"t [Gyr]")

plt.savefig(figure_Path("Sec4_3_2-time_profiles.png"))
console.print(f"[bold green]SUCCESS[/bold green] saved under: {figure_Path("Sec4_3_2-time_profiles.png")}\n")

plt.show()
