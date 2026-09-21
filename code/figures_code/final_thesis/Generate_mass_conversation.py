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

cfg_Gravothermal = Gravothermal_data_by_keys(("A"), ("dual",""))
cfg_Nbody = Nbody_data_by_keys(("A", "B"))
r_goal = 0.0775

fig, ax = plt.subplots(1, 1)


for cfg in cfg_Gravothermal:
    kwargs = get_kwargs(cfg)
    m_0 = cfg["m"][0,-1]
    m_arr = cfg["m"][:,-1]
    t = cfg["t"]/ cfg["tcc"][0]-cfg["tcmin"][0]/cfg["tcc"][0]
    
    
    ax.plot(t, m_arr/m_0, **kwargs)
for cfg in cfg_Nbody:
    kwargs = get_kwargs(cfg)
    m_0 = cfg["m_enclosed"][0,-1]
    m_arr = cfg["m_enclosed"][:,-1]
    err = np.clip(cfg["m_enclosed_err"][:,-1],0,10**(6.5))
    t = cfg["t"]/ cfg["tcc"][0] -cfg["tcmin"][0]/cfg["tcc"][0]

    ax.fill_between(
        t,
        (m_arr-err)/m_0,
        (m_arr+err)/m_0,
        alpha=STYLES["nbody_benchmark"]["alpha_fill"],
        edgecolors="black",
        color=kwargs["color"],
    )    
    
    ax.plot(t, m_arr/m_0, **kwargs)


import numpy as np
from scipy.interpolate import interp1d

x_nbody = cfg_Nbody[0]["t"]
x_nbody_scaled = x_nbody / cfg_Nbody[0]["tcc"][0]
y_nbody = cfg_Nbody[0]["rho_avrg"]

r_goal = 0.0775
t_fluid_raw = cfg_Gravothermal[0]["t"]
x_fluid_scaled = t_fluid_raw / cfg_Gravothermal[0]["tcc"][0]

y_fluid_raw = cfg_Gravothermal[0]["f_rho"](t_fluid_raw, np.full_like(t_fluid_raw, r_goal)).flatten()

value= []
for r, arr in zip(cfg_Gravothermal[0]["r"],cfg_Gravothermal[0]["rho"]):
    value.append(interp1d(r, arr, bounds_error=False, fill_value="extrapolate")(cfg_Nbody[0]["r_goal"]))

fluid_interpolator = interp1d(x_fluid_scaled, value, bounds_error=False, fill_value="extrapolate")
y_fluid_resampled = fluid_interpolator(x_nbody_scaled)

y_relative_diff = abs(y_nbody-y_fluid_resampled) / y_nbody

ax2 = ax.twinx()

line, = ax2.plot(x_nbody_scaled-cfg_Nbody[0]["tcmin"][0]/cfg_Nbody[0]["tcc"][0], y_relative_diff, label="...", color="lightgray", linewidth=3, zorder=-4)
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)
ax2.set_ylabel(r"$\Delta\rho/\rho$")


kw={"zorder":-5, "color":"black"}
ax.set_ylim(0.8,1.01)
ax.set_xlim(-0.2,1.12)

create_legend(
    ax, cfg_Nbody, cfg_Gravothermal, extra=[[line], [r"$\Delta \rho/\rho$"]], bbox=(0.0,0.0), loc="lower left", ADD_RUN=True
)

ax.set_xlabel(r"$\tau= (t-t_\text{cf})/t_\text{cc}$")
ax.set_ylabel(r"$M_\text{total}(t)/M_\text{total}(t=0)$")


ymin_ax, ymax_ax = ax.get_ylim()

y_target_ax = 1.0 
ymin_ax2 = 0.4   

scale = ymin_ax2 / (ymin_ax - y_target_ax)
ymax_ax2 = scale * (ymax_ax - y_target_ax)

ax2.set_ylim(ymin_ax2, ymax_ax2)

name="Sec4_3_2-mass_enclosed.png"
plt.savefig(figure_Path(name))
console.print(f"[bold green]SUCCESS[/bold green] saved under: {figure_Path(name)}\n")

plt.show()