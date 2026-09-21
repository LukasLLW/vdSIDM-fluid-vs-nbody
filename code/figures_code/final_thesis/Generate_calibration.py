import matplotlib.pyplot as plt
from data_path import figure_Path
import numpy as np
from rich.console import Console
from analyse_code.Moments import moments
from figures_code.final_thesis.generell_dics import (PALETTES,STYLES,create_legend,get_kwargs)
from figures_code.final_thesis.LOAD_GRAVOTHERMAL_DATA import (Gravothermal_data_by_keys, base)
from figures_code.final_thesis.LOAD_NBODY_DATA import Nbody_data_by_keys
from scipy.interpolate import CubicSpline
from pathlib import Path 
import re
from data_path import gravothermal_root
plt.style.use(r"./figures_code/style_format_one.mplstyle")

console = Console()

PATH_A, PATH_B = Path(r"G:\DATABASE\ZIP\sigma_m_2472_71"), Path(r"G:\DATABASE\ZIP\sigma_m_6593_89")

cfg_Gravothermal = Gravothermal_data_by_keys(("A", "B"), ("dual", ""))
cfg_Nbody = Nbody_data_by_keys((""))


fig_BA, ax_BA = plt.subplots(1,3, sharey=True)
axes_BA = {"n_shells": ax_BA[0], "r_min": ax_BA[1], "r_max": ax_BA[2]}

fig_check, ax_check = plt.subplots(1,3)
axes_check = {"n_shells": ax_check[0], "r_min": ax_check[1], "r_max": ax_check[2]}


for key, PATH_DIR in zip(["A","B"], [PATH_A, PATH_B]): 
    
    
    for variation in ["n_shells" ,"r_min","r_max"]:

        shells = [x for x in (PATH_DIR / f"variation_{variation}").iterdir() if x.is_dir()]

        cfg_var = []
        for PATH in shells:
            cfg = base(PATH, key=key)
            filename = PATH.name

            match = re.search(f"{variation}_(\d+)_(\d+)", filename)

            if match:
                float_str = f"{match.group(1)}.{match.group(2)}"
            else:
                match = re.search(f"{variation}_(\d+)", filename)
                float_str = f"{match.group(1)}"

            cfg[variation] = float(float_str)
            cfg_var.append(cfg)
            cfg_var.sort(key=lambda x: x[variation])

        [axes_check[variation].semilogy(cfg["t"], cfg["rho"][:,0], label=cfg[variation]) for cfg in cfg_var]
        [axes_check[variation].scatter(cfg["tcmin"][0], cfg["tcmin"][1]) for cfg in cfg_var]
        [axes_check[variation].scatter(cfg["tcc"][0], cfg["tcc"][1]) for cfg in cfg_var]

        data = [cfg["tcmin"][0] / cfg["tcc"][0] for cfg in cfg_var]
        VARIABL = [cfg[variation] for cfg in cfg_var]
                
        
        axes_BA[variation].set_xlabel(variation)
        axes_check[variation].set_xlabel(variation)
        axes_BA[variation].scatter(VARIABL, data, color = PALETTES[key][STYLES["dual"]["c"]])

        if len(VARIABL)>1:
            VARR = np.linspace(VARIABL[0], VARIABL[-1],100)
            VARIABL, indices = np.unique(VARIABL, return_index=True)
            data = np.array(data)[indices]

            spline = CubicSpline(VARIABL, data)
            axes_BA[variation].plot(VARR, spline(VARR), color = PALETTES[key][STYLES["dual"]["c"]], linestyle= STYLES["dual"]["linestyle"])
    

axes_BA["n_shells"].set_xlabel(r"$n$")
axes_BA["r_min"].set_xlabel(r"$r_\text{min}$")
axes_BA["r_max"].set_xlabel(r"$r_\text{max}$")

ax_BA[0].set_ylabel(r"$t_\text{cf} / t_\text{cc}$")

create_legend(
    ax_BA[0], cfg_Nbody, cfg_Gravothermal, [[], []], ADD_RUN=True, bbox=(0.98,0.00)
)
from matplotlib.ticker import ScalarFormatter

for var, ax in axes_BA.items():
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2)) 
    ax.xaxis.set_major_formatter(formatter)

ax_BA[0].set_ylim(0.12,0.145)
fig_BA.subplots_adjust(bottom=0.20, left=0.10, right=0.98, top=0.95)
name="Sec4_2_2-callibration.png"
fig_BA.savefig(figure_Path(name), bbox_inches="tight")
console.print(f"[bold green]SUCCESS[/bold green] saved under: {figure_Path(name)}\n")


plt.show()

