import analyse_code.lattice_boltzmann_approche as LB
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy import integrate
import matplotlib.pyplot as plt
from data_path import figure_Path
import numpy as np
from rich.console import Console
import matplotlib.gridspec as gridspec

from figures_code.final_thesis.generell_dics import (PALETTES, STYLES, create_legend, get_kwargs)
from figures_code.final_thesis.LOAD_GRAVOTHERMAL_DATA import (Gravothermal_data_by_keys,)
from figures_code.final_thesis.LOAD_NBODY_DATA import Nbody_data_by_keys
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm, SymLogNorm 
import seaborn as sns

plt.style.use(r".\figures_code\style_format_one.mplstyle")


plt.rcParams['xtick.color'] = '#333333'      
plt.rcParams['ytick.color'] = '#333333'      
plt.rcParams['axes.edgecolor'] = '#333333'  
plt.rcParams['text.color'] = '#333333'       
plt.rcParams['axes.labelcolor'] = '#333333'

console = Console()

cfg_Gravothermal = Gravothermal_data_by_keys(("B"), ("dual",""))
main_set = cfg_Gravothermal[0]

r_s = 1.28
rho_s = 4.4e7

r_arr = main_set["r"][0, 1:] / r_s
rho_arr = main_set["rho"][0, 1:] / rho_s

(edd_grid,
 EPSILON_GRID,
 velocity_grid,
 speed_distribution,
 probability_grid, escape_velo1
) = LB.f_Eddington(r_arr, rho_arr)

PSI_r, PSI_rho, escape_velo = LB.PHI_GRAVITATIONAL(r_arr, rho_arr)
probability_grid_mb = LB.f_Maxwell_Boltzmann(r_arr, rho_arr, velocity_grid, PSI_r)

edd_grid_mb = np.zeros_like(probability_grid_mb)
for i in range(len(r_arr)):
    v_mask = velocity_grid[i] > 0
    edd_grid_mb[i, v_mask] = probability_grid_mb[i, v_mask] / (4.0 * np.pi * velocity_grid[i, v_mask]**2)

fig = plt.figure(figsize=(7.5, 9.5))

gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 0.04], height_ratios=[1, 1, 1.2], figure=fig)

ax00 = fig.add_subplot(gs[0, 0])
ax10 = fig.add_subplot(gs[1, 0], sharex=ax00)
ax20 = fig.add_subplot(gs[2, 0])
ax01 = fig.add_subplot(gs[0, 1], sharey=ax00)
ax11 = fig.add_subplot(gs[1, 1], sharex=ax01, sharey=ax10)
ax21 = fig.add_subplot(gs[2, 1])

cax0 = fig.add_subplot(gs[0, 2])
cax1 = fig.add_subplot(gs[1, 2])

axes = np.array([[ax00, ax01],
                 [ax10, ax11],
                 [ax20, ax21]])

f_plot_edd = np.where(edd_grid > 1e-12, edd_grid, 1e-12)
f_plot_mb = np.where(edd_grid_mb > 1e-12, edd_grid_mb, 1e-12)
p_plot_edd = np.where(probability_grid > 1e-6, probability_grid, 1e-6)
p_plot_mb = np.where(probability_grid_mb > 1e-6, probability_grid_mb, 1e-6)

R_GRID = np.zeros_like(velocity_grid)
for i in range(len(r_arr)):
    R_GRID[i, :] = r_arr[i]

fmax_shared = max(f_plot_edd.max(), f_plot_mb.max())
pmax_shared = max(p_plot_edd.max(), p_plot_mb.max())

Color_Map_1 = sns.color_palette("rocket", as_cmap=True)
Color_Map_2 = sns.color_palette("mako", as_cmap=True)

im00 = axes[0, 0].pcolormesh(R_GRID, velocity_grid / escape_velo[:, None], f_plot_edd, cmap=Color_Map_1, 
                             norm=LogNorm(vmin=1e-6, vmax=fmax_shared), shading='auto')

im01 = axes[0, 1].pcolormesh(R_GRID, velocity_grid / escape_velo[:, None], f_plot_mb, cmap=Color_Map_1, 
                             norm=LogNorm(vmin=1e-6, vmax=fmax_shared), shading='auto')



im10 = axes[1, 0].pcolormesh(R_GRID, velocity_grid / escape_velo[:, None], p_plot_edd, cmap=Color_Map_2, 
                             norm=LogNorm(vmin=1e-3, vmax=pmax_shared), shading='auto')

im11 = axes[1, 1].pcolormesh(R_GRID, velocity_grid / escape_velo[:, None], p_plot_mb, cmap=Color_Map_2, 
                             norm=LogNorm(vmin=1e-3, vmax=pmax_shared), shading='auto')

fig.colorbar(im01, cax=cax0, label=r"$f(\varepsilon)$")
fig.colorbar(im11, cax=cax1, label=r"$P(v)$")

def normalize_density(x, y, is_3d_distribution=False):
    """
    Normiert ein Profil y(x). 
    Falls is_3d_distribution=True, wird der 3D-Phasenraumfaktor (4 * pi * x^2) 
    bei der Integration berücksichtigt.
    """
    if is_3d_distribution:
        integrand = 4.0 * np.pi * (x**2) * y
        area = integrate.simpson(y=integrand, x=x)
    else:
        area = integrate.simpson(y=y, x=x)
        
    return y / area if area > 0 else y

idx_min = 0
r_min_val = r_arr[idx_min]
x_min = velocity_grid[idx_min] / escape_velo[idx_min, None]
y_min_edd = normalize_density(x_min, probability_grid[idx_min], is_3d_distribution=False)
y_min_mb = normalize_density(x_min, probability_grid_mb[idx_min], is_3d_distribution=False)



norm_edd_min = sum(probability_grid[idx_min])
norm_mb_min = sum(probability_grid_mb[idx_min])

axes[2, 0].plot(x_min, probability_grid[idx_min]/norm_edd_min, color='darkred', lw=2.5, label=fr'Eddington ($r/r_s \approx {r_min_val:.2f}$)')
axes[2, 0].plot(x_min, probability_grid_mb[idx_min]/norm_mb_min, color='salmon', lw=2, ls='--', label=fr'MB ($r/r_s \approx {r_min_val:.2f}$)')

idx_mid = int(len(r_arr) * 0.4)
r_mid_val = r_arr[idx_mid]
x_mid = velocity_grid[idx_mid] / escape_velo[idx_mid, None]

norm_edd_mid=sum(probability_grid[idx_mid])
norm_mb_mid=sum(probability_grid_mb[idx_mid])

axes[2, 0].plot(x_mid, probability_grid[idx_mid]/norm_edd_mid, color='teal', lw=2.5, label=fr'Eddington ($r/r_s \approx {r_mid_val:.2f}$)')
axes[2, 0].plot(x_mid, probability_grid_mb[idx_mid]/norm_mb_mid, color='chartreuse', lw=2, ls='--', label=fr'MB ($r/r_s \approx {r_mid_val:.2f}$)')

p_val = 3.0
w_val = 0.31978

Kp_profile_edd = []
Kp_profile_mb = []

for idx in range(len(r_arr)):
    k5_edd = LB.compute_Kp_fast(idx, edd_grid, velocity_grid, p_val, w_val)
    k5_mb = LB.compute_Kp_fast(idx, edd_grid_mb, velocity_grid, p_val, w_val)
    
    Kp_profile_edd.append(k5_edd)
    Kp_profile_mb.append(k5_mb)

sm= 6593.89
axes[2, 1].plot(r_arr[:-3]*r_s, sm * np.array(Kp_profile_edd[:-3]), label="Eddington", color="darkblue")
axes[2, 1].plot(r_arr[:-3]*r_s, sm * np.array(Kp_profile_mb[:-3]), label="Maxwell-Boltzmann", color="orange")
axes[2, 1].set_yscale("log")
ax00.set_xscale("log")
ax01.set_xscale("log")

ax21.set_xscale("log")

axes[0, 0].set_title(r"Eddington $f(v|r)$")
axes[0, 1].set_title(r"Maxwell-Boltzmann $f(v|r)$")
axes[1, 0].set_title(r"Eddington $P(v|r)$")
axes[1, 1].set_title(r"Maxwell-Boltzmann $P(v|r)$")

axes[0, 0].set_ylabel(r"$v / v_{\mathrm{esc}}$")
axes[0, 1].set_ylabel(r"$v / v_{\mathrm{esc}}$")
axes[1, 0].set_ylabel(r"$v / v_{\mathrm{esc}}$")
axes[1, 1].set_ylabel(r"$v / v_{\mathrm{esc}}$")
axes[2, 0].set_ylabel(r"$f_\text{normed}(v)$")

axes[2, 1].yaxis.tick_right()
axes[2, 1].set_ylabel(r"$\langle \sigma_v \rangle \quad \Big[\text{cm}^2 / \text{g} \Big]$")
axes[1, 0].set_xlabel(r"$r / r_s$")
axes[0, 0].set_xlabel(r"$r / r_s$")
axes[1, 1].set_xlabel(r"$r / r_s$")
axes[0, 1].set_xlabel(r"$r / r_s$")
axes[2, 0].set_xlabel(r"$v / v_{\mathrm{esc}}$")
axes[2, 1].set_xlabel(r"$r \quad [\text{kpc}]$")

axes[2, 0].legend()
axes[2, 1].legend()


axes[2, 1].yaxis.tick_right()
axes[2, 1].yaxis.set_label_position("left")
axes[2, 1].tick_params(
    axis='y',
    which='both',
    left=True,
    labelleft=False,
    right=True,
    labelright=True
)
plt.tight_layout()

ax00.tick_params(labelbottom=False)
ax01.tick_params(labelbottom=False)

ax01.tick_params(labelleft=False)
ax11.tick_params(labelleft=False)

for ax in axes.flat:
    ax.tick_params(colors="#707070", which='both')
name="Sec4_3_5-Eddington_set_up.png"
plt.savefig(figure_Path(name))
console.print(f"[bold green]SUCCESS[/bold green] saved under: {figure_Path(name)}\n")



plt.show()