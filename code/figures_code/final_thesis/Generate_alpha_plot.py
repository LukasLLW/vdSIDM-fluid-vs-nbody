import matplotlib.pyplot as plt
from data_path import figure_Path
import numpy as np
from rich.console import Console
from analyse_code.Moments import moments
from figures_code.final_thesis.generell_dics import (PALETTES, STYLES, create_legend, get_kwargs)
from figures_code.final_thesis.LOAD_GRAVOTHERMAL_DATA import (Gravothermal_data_by_keys, base)
from figures_code.final_thesis.LOAD_NBODY_DATA import Nbody_data_by_keys
from scipy.interpolate import CubicSpline
import analyse_code.prepare_C_alpha_analysis as any
from pathlib import Path 
import re
import pandas as pd

plt.style.use(r".\figures_code\style_format_one.mplstyle")

console = Console()
console.print("[bold red] CHANGE PATH [/bold red]")


PATH_A, PATH_B = Path(r"C:\Users\lukas\Documents\GitHub_new26\DATABASE\ZIP\unpacked\sigma_m_2472_71"), Path(r"C:\Users\lukas\Documents\GitHub_new26\DATABASE\ZIP\unpacked\sigma_m_6593_89")


MIN_RHO_B = 93206275.00353076
T_CC_B = 6.525050100200401

fig_BA, (ax1_left, ax2_left) = plt.subplots(1, 2, sharey=True)
ax1_right = ax1_left.twinx()
ax2_right = ax2_left.twinx()


ax1_right.set_zorder(ax1_left.get_zorder() - 1)
ax2_right.set_zorder(ax2_left.get_zorder() - 1)
ax1_left.patch.set_visible(False)  
ax2_left.patch.set_visible(False)

fig_check, ax_check = plt.subplots(1, 2)

cfg_var = []
for key, PATH_DIR in zip(["A"], [PATH_A]): 
    PATH_dic = [
        next(sub.iterdir()) 
        for sub in PATH_DIR.glob("grid_point_*") 
        if sub.is_dir()
    ]
    
    for PATH in PATH_dic:
        filename = PATH.name
        cfg = base(PATH, key=key)
        cfg["path"] = PATH
        
        for variable in ["C", "alpha"]:
            match = re.search(f"{variable}_(\d+)_(\d+)", filename)
            if match:
                float_str = f"{match.group(1)}.{match.group(2)}"
            else:
                match = re.search(f"{variable}_(\d+)", filename)
                float_str = f"{match.group(1)}"
            
            cfg[variable] = float(float_str)
            
        if cfg["C"] < 0.33 or cfg["alpha"] < 0.3:
            continue

        cfg_var.append(cfg)

df = pd.DataFrame(cfg_var)

unique_c = np.sort(df["C"].unique())
unique_alpha = np.sort(df["alpha"].unique())

unique_c_reduced = unique_c[::2]
unique_alpha_reduced = unique_alpha[::2]

Trans1 = PALETTES["Transition 1"](len(unique_c_reduced))
Trans2 = PALETTES["Transition 2"](len(unique_alpha_reduced))

for i, C in enumerate(unique_c_reduced):
    df_filtered_c = df[df["C"] == C].sort_values("alpha")

    x = df_filtered_c["alpha"]
    y = [np.array(val)[1] for val in df_filtered_c["tcmin"]]
    y2 = [np.array(val)[0] for val in df_filtered_c["tcc"]]
    
    ax1_right.plot(x, np.array(y2) / T_CC_B, linestyle="--", color=Trans1[i], alpha=0.7, zorder=1)
    ax1_left.plot(x, np.array(y) / MIN_RHO_B, label=f"C = {C:.2f}", color=Trans1[i], zorder=3)

for i, ALPHA in enumerate(unique_alpha_reduced):
    df_filtered_alpha = df[df["alpha"] == ALPHA].sort_values("C")

    x = df_filtered_alpha["C"]
    y = [np.array(val)[1] for val in df_filtered_alpha["tcmin"]]
    y2 = [np.array(val)[0] for val in df_filtered_alpha["tcc"]]
    
    ax2_right.plot(x, np.array(y2) / T_CC_B, linestyle="--", color=Trans2[i], alpha=0.7, zorder=1)
    ax2_left.plot(x, np.array(y) / MIN_RHO_B, label=rf"$\alpha$ = {ALPHA:.2f}", color=Trans2[i], zorder=3)

ax1_left.set_xlabel(r"$\alpha$")
ax1_left.set_ylabel(r"$\rho_\text{cf}/ \rho_\text{cf, N-\text{body}}$", color="black")

ax1_right.tick_params(labelright=False)
ax1_right.set_ylabel("")

ax2_left.set_xlabel("C")
ax2_left.set_ylabel("") 

ax2_right.set_ylabel(r"$tcc / T_{CC, \, N-\text{body}}$ (dotted)", color="black")

leg1 = ax1_left.legend(title="C-Values", loc="upper right", fontsize="small", facecolor="white", framealpha=1.0, frameon=True)
leg2 = ax2_left.legend(title=r"$\alpha$-Values", loc="upper right",bbox_to_anchor=(0.98, 0.85), fontsize="small", facecolor="white", framealpha=1.0, frameon=True)

leg1.set_zorder(10)
leg2.set_zorder(10)

plt.tight_layout()

name = "Sec4_3_4-Alpha_Und_C.png"
fig_BA.savefig(figure_Path(name))
console.print(f"[bold green]SUCCESS[/bold green] saved under: {figure_Path(name)}\n")
plt.show()