import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from data_path import figure_Path
from figures_code.final_thesis.generell_dics import (
    PALETTES,
    STYLES,
    create_legend,
    get_kwargs,
)
from figures_code.final_thesis.LOAD_GRAVOTHERMAL_DATA import (
    Gravothermal_data_by_keys,
)
from figures_code.final_thesis.LOAD_NBODY_DATA import Nbody_data_by_keys
from matplotlib.lines import Line2D
from rich.console import Console
import matplotlib.patheffects as patheffects

console = Console()
plt.style.use(r".\figures_code\style_format_one.mplstyle")

cfg_Gravothermal = Gravothermal_data_by_keys(("B"), ("dual", "eff", "rescaling", "num"))
Benchmark_func = cfg_Gravothermal[0]["f_rho"]

v_min, v_max = float("inf"), float("-inf")
diff_data_list = []

t_bench, r_bench = cfg_Gravothermal[0]["t"], cfg_Gravothermal[0]["r"][0]

for i, cfg in enumerate(cfg_Gravothermal[1:]):
    t, r = cfg["t"], cfg["r"][0]
    t_m = t_bench < t[-1]
    mask = (6e-3 < r) & (r < 1e1)
    T, R = np.meshgrid(t_bench[t_m], r_bench)

    B = Benchmark_func(T.ravel(), R.ravel())
    F = cfg["f_rho"](T.ravel(), R.ravel())

    diff = (((F - B) / F) * 100).reshape(T.shape)
    diff_data_list.append((t_bench[t_m], r_bench, T, R, diff, cfg["model"],[cfg["t"], cfg["r"][:,0]]))

    v_min = min(v_min, np.nanmin(diff))
    v_max = max(v_max, np.nanmax(diff))

CMAP = sns.color_palette("Spectral_r", as_cmap=True)

norm = mcolors.SymLogNorm(
    linthresh=10.0, linscale=1.2, vmin=-100.0, vmax=100.0, base=10
)

max_times_dict = {data[5]: np.max(data[0]) for data in diff_data_list}
widths = [max_times_dict["eff"], max_times_dict["rescaling"], max_times_dict["num"]]

fig, ax = plt.subplots(
    1, 3, sharey=True, constrained_layout=True, gridspec_kw={"width_ratios": widths}
)

axes_dict = {"eff": ax[0], "rescaling": ax[1], "num": ax[2]}

title_dict = {
    "eff": r"$\sigma_\text{eff}$ vs. $K_5$-Analy.",
    "rescaling": r"Rescaled vs. $K_5$-Analy.",
    "num": "$K_5$-Num. vs $K_5$-Analy.",
}

for t, r_masked, T, R, diff, name, lim in diff_data_list:
    im = axes_dict[name].pcolormesh(
        T, R, diff, cmap=CMAP, norm=norm, shading="auto"
    )
    axes_dict[name].hlines(0.0775, 0, 8, color="black")
    axes_dict[name].set_title(title_dict[name])
    print(title_dict[name])
    axes_dict[name].set_xlim(0, np.max(t))

for t, r_masked, T, R, diff, name, lim in diff_data_list:
    im = axes_dict[name].pcolormesh(
        T, R, diff, cmap=CMAP, norm=norm, shading="auto"
    )
    axes_dict[name].set_xlim(0, np.max(t))
    axes_dict[name].plot(lim[0], lim[1], color="black")

    axes_dict[name].fill_between(
        lim[0], lim[1], color="lightgray", hatch="//", edgecolor="gray", alpha=0.5
    )

    txt = axes_dict[name].text(
        0.5,
        0.1,
        r"$r<r_{\mathrm{min}} \rightarrow \mathrm{extrapolated}$",
        color="black",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        transform=axes_dict[name].transAxes,
    )

for axes in ax:
    axes.set_xlabel(r"t [\text{Gyr}]")
    axes.set_yscale("log")

ax[0].set_ylabel(r"$r \quad [\text{kpc}]$")

cbar = fig.colorbar(
    im, 
    ax=ax, 
    orientation="vertical", 
    fraction=0.046, 
    pad=0.04,
    ticks=[-100, -50, -10, 0, 10, 50, 100]
)
cbar.set_label(r"$100 \cdot [\rho_{K_5 \text{-Analy.}} - \rho_\text{Model}]/ \rho_{K_5 \text{-Analy.}} \ [\pm \%]$")

fig.suptitle(
    "Deviations from $K_5$-Analy. (B):", 
    x=0.01, 
    y=1.045, 
    ha="left", 
    weight="bold"
)

txt = ax[0].text(
    0.2,
    0.085,
    r"$r_\text{core}$",
    color="black",
    fontsize=10,
    fontweight="bold",
    ha="left",
    va="bottom"
)


name="Sec4_3_3-model_diff.png"
plt.savefig(figure_Path(name))
console.print(f"[bold green]SUCCESS[/bold green] saved under: {figure_Path(name)}\n")

plt.show()