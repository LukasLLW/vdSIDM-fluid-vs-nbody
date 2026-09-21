import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple

PALETTES = {
    "A": sns.cubehelix_palette(
        n_colors=5,
        start=0.5,
        rot=-0.8,
        light=0.85,
        dark=0.4,
        hue=1.5,
        reverse=True,
    ),
    "B": sns.cubehelix_palette(
        n_colors=5,
        start=2.8,
        rot=0.8,
        light=0.65,
        dark=0.2,
        hue=1.8,
        reverse=True,
    ),
    "C": ["#b87903"], 
    "Analytic": sns.cubehelix_palette(n_colors=2, start=0.0, rot=-0.1, light=0.1, dark=0.9),
    "Transition 1": lambda n: sns.color_palette("viridis", n_colors=n,),
    "Transition 2": lambda n: sns.color_palette("rocket", n_colors=n),
    "Diff": sns.color_palette("icefire", as_cmap=True)
}

STYLES = {
    "nbody_benchmark": {
        "linestyle": "-",
        "linewidth": 2,
        "alpha_fill": 0.15,
        "label_name": "Model",
        "zorder": -2,
        "c": 0
    },
    "num": {
        "linestyle": "-",
        "linewidth": 2.8,
        "zorder": -1,
        "label_name": "num.",
        "c": 1
    },
    "eff": {
        "linestyle": "-.",
        "linewidth": 1.5,
        "zorder": 3,
        "label_name": "eff.",
        "c": 2
    },
    "rescaling": {
        "linestyle": ":",
        "linewidth": 2,
        "zorder": 2,
        "label_name": "rescale",
        "c": 4
    },
    "dual": {
        "linestyle": "--",
        "linewidth": 2,
        "zorder": 4,
        "label_name": "disp.-dep.",
        "c": 3
    },
    "base": {
        "linestyle": "-",
        "linewidth": 2,
        "zorder": 4,
        "label_name": "Dual",
        "c": 4
    },
}
def get_kwargs(cfg):
    if cfg["style"] not in STYLES.keys():
        style = STYLES["base"]
    else:
        style = STYLES[cfg["style"]]
    return {
            "linestyle": style["linestyle"], 
            "linewidth": style["linewidth"], 
            "color": PALETTES[cfg["run"]][style["c"]] if cfg["run"] in PALETTES.keys() else "red",
            "zorder": style["zorder"]
        }
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

def create_flexible_inplot_legend(ax_target, handles, labels, loc="lower right", bbox=(0.99, 0.02)):

    clean_handles = []
    clean_labels = []
    
    for h, l in zip(handles, labels):
        if l and l.strip():

            if not isinstance(h, tuple) and hasattr(h, "set_markeredgewidth"):
                h.set_markeredgewidth(0.8)
                h.set_markeredgecolor("#1A1A1A")  
                h.set_markerfacecolor("#8A8A8A")
            clean_handles.append(h)
            clean_labels.append(l)

    leg = ax_target.legend(
        handles=clean_handles,
        labels=clean_labels,
        loc=loc,
        bbox_to_anchor=bbox,
        frameon=False,
        labelspacing=0.4,
        handletextpad=0.6,
        fontsize=9
    )

    return leg

def create_flexible_inplot_legend_gtouped(ax_target, handles_group, labels_group, titles=["N-body", "fluid", "other"], loc="lower right", bbox=(0.99, 0.02),space=0.05):
    """
    Creates a clean, frameless legend from raw handles and labels.
    Supports tuples of handles (e.g., combining a patch and a line for N-Body data).
    """
    x= bbox[0]
    
    y=bbox[1] + space*len(labels_group)
    for handels, labels, title, i in zip(handles_group,labels_group, titles, range(len(labels_group))):
        if title:
                title_handle = mpatches.Rectangle((0, 0), 0, 0, fill=False, edgecolor='none', visible=False)
                title_label = f"$\\textbf{{{title}}}$" 
                handels.insert(0, title_handle)
                labels.insert(0, title_label)        
        leg= ax_target.legend(
            handles=handels,
            labels=labels,
            ncol=4,
            columnspacing=0.5,
            loc=loc,
            bbox_to_anchor=(x,y-space*i),
            frameon=False,
            labelspacing=0.2,
            handletextpad=0.6,
            fontsize=9,
        )
        ax_target.add_artist(leg)
    return leg



def create_legend(ax_A, nbody, gravothermal_fluid, extra, loc="lower right", bbox=(0.99, 0.02), ADD_RUN=False,space=0.05):
    """
    Specific convenience function that maps configs to styles, adds an error-shading
    patch behind the N-Body line icon, and passes them to the flexible base function.
    """
    LEGEND_MAPPING = {
        "nbody_benchmark": "N-Body",
        "num":  "$K_5$-Num.",
        "dual": "$K_5$-Analy.",
        "eff": r"$\sigma_\text{eff}$ ",
        "rescaling": "Rescaled",
        "base": "Base Model",
    }

    all_handles = []
    all_labels = []
    grouped = {}
    nbody_handels = []
    nbody_labels = []
    for cfg in nbody:
        style = "nbody_benchmark"

        if style not in grouped or ADD_RUN:
            kwargs = get_kwargs(cfg)
            kwargs["color"] = PALETTES[cfg["run"]][STYLES[style].get("c", "red")] if ADD_RUN else "#4A4A4A" 
                        

            base_name = LEGEND_MAPPING.get(style, STYLES[style].get("label_name", style))
            display_name = base_name

            line_obj = mlines.Line2D([], [], **kwargs)
            
            error_patch = mpatches.Patch(color="#4A4A4A", alpha=0.18, linewidth=0)
            
            all_handles.append((error_patch, line_obj))
            nbody_handels.append((error_patch, line_obj))
            
            all_labels.append(display_name)
            nbody_labels.append(cfg["run"])
            grouped[style] = True

    if nbody_labels:
        sorted_pairs = sorted(zip(nbody_labels, nbody_handels))
        
        nbody_labels, nbody_handels = zip(*sorted_pairs)
        
        nbody_labels = list(nbody_labels)
        nbody_handels = list(nbody_handels)

        for i in range(len(nbody_labels) - 1):
            nbody_labels[i] = f"{nbody_labels[i]}, "

    LookUp = {"dual": "$K_5$-Analy.","eff": r"$\sigma_\text{eff}$", "num":  "$K_5$-Num.", "rescaling": "Rescaled"}
    styl_handels, styl_labels = {}, {}
    title_fluid = []
    for cfg in gravothermal_fluid:
        style = cfg["style"]
        kwargs = get_kwargs(cfg)
        kwargs["color"] = PALETTES[cfg["run"]][STYLES[style].get("c", "red")] if ADD_RUN else "#4A4A4A" 
            
        display_name = LEGEND_MAPPING.get(
            style, STYLES[style].get("label_name", style)
        ) if ADD_RUN is False else f"{LEGEND_MAPPING.get(
            style, STYLES[style].get("label_name", style),
        )}"
        
        
        line_obj = mlines.Line2D([], [], **kwargs)


        if style not in grouped:
            title_fluid.append(fr"{LookUp[style]}")
            all_handles.append(line_obj)
            styl_handels[style] = [line_obj]
            all_labels.append(display_name)
            styl_labels[style] = [cfg["run"]]
            grouped[style] = True
        elif ADD_RUN:
            styl_handels[style].append(line_obj)
            styl_labels[style].append(cfg["run"])

    fluid_handels = []
    fluid_labels = []

    for style in styl_labels.keys():
        handles_for_style = styl_handels[style]
        labels_for_style = styl_labels[style]
        
        sorted_pairs = sorted(zip(labels_for_style, handles_for_style))
        labels_for_style, handles_for_style = zip(*sorted_pairs)
        labels_for_style = list(labels_for_style)
        handles_for_style = list(handles_for_style)
        
        for i in range(len(labels_for_style) - 1):
            labels_for_style[i] = f"{labels_for_style[i]},"
            
        fluid_handels.append(handles_for_style)
        fluid_labels.append(labels_for_style)

    extra_handles, extra_labels = extra
    for h, l in zip(extra_handles, extra_labels):
        all_handles.append(h)
        all_labels.append(l)
    if ADD_RUN:
        final_handles = []
        final_labels = []
        final_titles = []

        if nbody_handels:
            final_handles.append(nbody_handels)
            final_labels.append(nbody_labels)
            final_titles.append("N-body")

        if fluid_handels:
            for h_sub, l_sub, t_sub in zip(fluid_handels, fluid_labels, title_fluid):
                if h_sub:  
                    final_handles.append(h_sub)
                    final_labels.append(l_sub)
                    final_titles.append(t_sub)

        if extra and len(extra[0]) > 0:
            final_handles.append(extra[0])
            final_labels.append(extra[1])
            final_titles.append("Analysis")

        if not final_handles:
            return None

        return create_flexible_inplot_legend_gtouped(
            ax_A, 
            final_handles, 
            final_labels, 
            titles=final_titles, 
            loc=loc, 
            bbox=bbox, 
            space=space
        )
    else:
        return create_flexible_inplot_legend(ax_A, all_handles, all_labels, loc=loc, bbox=bbox)