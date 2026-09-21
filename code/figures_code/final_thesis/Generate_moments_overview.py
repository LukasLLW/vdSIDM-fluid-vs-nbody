import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import exp1
from scipy.integrate import quad
from analyse_code.Moments import moments, visco
from figures_code.final_thesis.generell_dics import PALETTES, create_flexible_inplot_legend_gtouped
import matplotlib.colors as mcolors
import colorsys
import seaborn as sns
from data_path import figure_Path
from rich.console import Console
from figures_code.final_thesis.LOAD_GRAVOTHERMAL_DATA import (Gravothermal_data_by_keys, base)
cfg_Gravothermal = Gravothermal_data_by_keys(("A","B"), ("dual", ""))
console = Console()
plt.style.use(r'.\figures_code\style_format_one.mplstyle')
from figures_code.style_rules import plot_styles
from data_path import data_root
CSV_DIR = data_root("viscosity_moment_csv")
OUTPUT_FILE = "viscosity_moments_overview.png"

R_S = 1.28
RHO_S = 0.044175
G = 1.0 
V_MAX = 1.649 * R_S * np.sqrt(G * RHO_S)
SIGMA_EFF_1D = V_MAX / np.sqrt(3.0)


def read_csv_file(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = data.dtype.names
    return data[names[0]], data[names[1]]


def extract_moment_from_filename(path):
    stem = path.stem
    raw = stem.split("_")[-1]
    return float(raw.replace("p", "."))


def contrast_color(rgba_color):
    r, g, b, a = rgba_color

    h, l, s = colorsys.rgb_to_hls(r, g, b)

    h_highlight = (h + 0.5) % 1.0
    
    s = min(s * 1.3, 1.0)
    
    if l > 0.7:
        l = 0.4
    elif l < 0.3:
        l = 0.6

    r_high, g_high, b_high = colorsys.hls_to_rgb(h_highlight, l, s)
    return (r_high, g_high, b_high, a)


def main():
    csv_files = sorted(
        CSV_DIR.glob("*.csv"),
        key=extract_moment_from_filename,
    )

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {CSV_DIR}")

    internal_moments = np.array([
        extract_moment_from_filename(csv_file)
        for csv_file in csv_files
    ])
    moments_numeric = internal_moments - 2

    fig, ax = plt.subplots()
    cmap = sns.color_palette("icefire", as_cmap=True)
    norm = plt.Normalize(vmin=np.min(moments_numeric), vmax=np.max(moments_numeric))

    for csv_file, display_moment in zip(csv_files, moments_numeric):
        x, y = read_csv_file(csv_file)
        ax.semilogx(
            x, y,
            linewidth=1,
            alpha=0.6,
            color=cmap(norm(display_moment)),
            zorder=1
        )

    x_ref = np.logspace(-3, 3, 120)
    
    color_3 = contrast_color(cmap(norm(3.0)))
    color_5 = contrast_color(cmap(norm(5.0)))
    color_7 = contrast_color(cmap(norm(7.0)))

    analytic_specs = [
        ("Fischer 3", "Order 3", color_3),  
        ("Fischer 5", "Order 5", color_5),  
        ("Fischer 7", "Order 7", color_7),  
    ]
    
    handles = []

    fischer_5_func = None
    for model_name, label, color in analytic_specs:
        func = moments(model_name)
        if "5" in model_name:
            fischer_5_func = func
            
        y_ref = func(x_ref)
        
        line, = ax.semilogx(
            x_ref, y_ref,
            linestyle="--", 
            linewidth=1.5,    
            color=color, 
            label=label,
            zorder=3
        )
        handles.append(line)
        
    line_visco, = ax.semilogx(
        x_ref, visco(x_ref),
        linestyle="-", 
        linewidth=1.25,
        color="black",
        label=r"Viscosity $\sigma_v$",
        zorder=2
    )
    handles.append(line_visco)   

    if fischer_5_func is not None:

        y_point = fischer_5_func(SIGMA_EFF_1D)
        print(cfg_Gravothermal[0]["Velo_disp [km/s]"]/20, cfg_Gravothermal[0]["sigma_eff [cm^2/g]"]/cfg_Gravothermal[0]["sigma_0"], 
              cfg_Gravothermal[0]["sigma_eff [cm^2/g]"])

        point_handle, = ax.plot(
             cfg_Gravothermal[0]["Velo_disp [km/s]"]/20, cfg_Gravothermal[0]["sigma_eff [cm^2/g]"]/cfg_Gravothermal[0]["sigma_0"],
            marker="*",
            markersize=10,
            color="#ff00ff",    
            markeredgecolor="black",
            markeredgewidth=0.7,
            linestyle="None",
            label=r"$\sigma_{\mathrm{eff}}$ Scale",
            zorder=5
        )

        handles.append(point_handle)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.03, shrink=0.85)
    
    ticks = np.linspace(moments_numeric.min(), moments_numeric.max(), 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{tick:.1f}" for tick in ticks])
    cbar.set_label(r"Moment $p$", rotation=270, labelpad=12)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)

    ax.set_xlabel(r"$v_\text{rel} / w$")
    ax.set_ylabel(r"$\langle \sigma \rangle / \sigma_0$")
    ax.set_xlim(1e-2, 1e1)
    ax.set_ylim(-0.05, 1.05)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.tick_params(width=0.6, labelsize=8)

    ax.legend(loc="best", fontsize=8)
    create_flexible_inplot_legend_gtouped(ax,[handles,[line_visco, point_handle]], [["3d", "5th", "7th"],[r"$\sigma_\text{visc}(v_\text{rel}/w) $", r"$\sigma_\text{eff} / \sigma_0$"]], ["Moments:", "Analysis:"], "upper right",(0.98, 0.88))

    from data_path import figure_Path
    name="Sec4_1_3-Moments_Overview.png"
    plt.savefig(figure_Path(name))
    console.print(f"[bold green]SUCCESS[/bold green] saved under: {figure_Path(name)}\n")


    plt.show()

    print(f"Plot successfully saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()