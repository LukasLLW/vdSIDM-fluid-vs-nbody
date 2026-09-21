import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from data_path import figure_Path
import numpy as np
from rich.console import Console

from figures_code.final_thesis.generell_dics import (PALETTES,STYLES,create_legend,get_kwargs)
from figures_code.final_thesis.LOAD_GRAVOTHERMAL_DATA import (Gravothermal_data_by_keys,)
from figures_code.final_thesis.LOAD_NBODY_DATA import Nbody_data_by_keys
from matplotlib.lines import Line2D

console = Console()
plt.style.use(r".\figures_code\style_format_one.mplstyle")

cfg_Gravothermal = Gravothermal_data_by_keys(("A", "B"), ("dual", "num", "eff", "rescaling"))
cfg_Nbody = Nbody_data_by_keys(("A", "B"))
r_goal = 0.0775

fig, ax = plt.subplots(1, 1)

ax_ins = ax.inset_axes([0.1, 0.1, 0.35, 0.35])

for i, cfg in enumerate(cfg_Gravothermal):
    kwargs = get_kwargs(cfg)
    
    idx_min = cfg["tcf_idx"]    
    x = cfg["r"][idx_min]
    y = cfg["rho"][idx_min,:]
    ax.loglog(x,y, **kwargs)
    ax_ins.loglog(x,y, **kwargs)

for i, cfg in enumerate(cfg_Nbody):
    kwargs = get_kwargs(cfg)
    
    idx_min = cfg["tcf_idx"]
    x = cfg["r"][3:]
    y = cfg["rho"][idx_min,3:]
    
    mask = y > 1e4

    x=x[mask]
    y=y[mask]
    
    err=np.clip(cfg["rho_err"][idx_min,3:][mask],0,10**(7.4))
    ax.loglog(x,y, **kwargs)
    ax_ins.loglog(x,y, **kwargs)
    ax.fill_between(
        x,
        y - err,
        y + err,
        alpha=STYLES["nbody_benchmark"]["alpha_fill"],
        edgecolors="black",
        color=kwargs["color"],
    )
    ax_ins.fill_between(
        x,
        y - err,
        y + err,
        alpha=STYLES["nbody_benchmark"]["alpha_fill"],
        edgecolors="black",
        color=kwargs["color"],
    )

x1, x2 = 3.5e-2, 9e-1
y1, y2 = 6.5e7, 1.5e8
ax_ins.set_xlim(x1, x2)
ax_ins.set_ylim(y1, y2)
ax.set_xlim(0.008, 2e1)
ax_ins.set_xscale('log')
ax_ins.set_yscale('log')

ax_ins.tick_params(axis='both', which='both', labelsize=8) 
ax.indicate_inset_zoom(ax_ins, edgecolor="black", alpha=0.3)

ax.set_xlabel(r"$r \quad [\text{kpc}]$")
ax.set_ylabel(r"$\rho \quad \Big [\text{M}_\odot / \text{kpc}^3 \Big]$")

create_legend(
    ax, cfg_Nbody, cfg_Gravothermal, extra=[[], []], bbox=(0.98,0.6), loc="lower right", ADD_RUN=True
)

r_goal = 0.0775
ymin, ymax = 0, 1e11
y_text_pos = 1.2e8

ax_ins.vlines(r_goal, ymin, ymax, colors='lightgray', linestyles='dashed')

ax_ins.annotate(
    r'$r_{\text{core}}$ Evaluation', 
    xy=(r_goal, y_text_pos), 
    xytext=(5, 0),              
    textcoords="offset points", 
    va='center',                
    ha='left',                  
    rotation=0              
)

plt.savefig(figure_Path("Sec4_3_2-radius_profiles_tf.png"))
console.print(f"[bold green]SUCCESS[/bold green] saved under: {figure_Path("Sec4_3_2-radius_profiles_tf.png")}\n")


plt.show()