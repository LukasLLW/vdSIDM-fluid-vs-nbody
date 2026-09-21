import numpy as np
from analyse_code.Gravothermal_interpolation import get_interpolation
from analyse_code.sigma_eff import sigma_eff
from astropy import units as ut
from analyse_code.Profile_Time_Rescalin import new_t
from reader_code.gravothermal_reader_code.read_Nishikawa_Data import (
    get_scaled_halo_dataset,
)
import os
from pathlib import Path
from rich.console import Console
from scipy.interpolate import CubicSpline
console = Console()

eta = 1.05
r_goal = 0.0775
current_path = os.path.abspath(__file__).replace("\\", "/")
code_index = current_path.rfind("/code/")
MAIN_DIR = Path(current_path[:code_index])
DATA_DIR_A= MAIN_DIR / "database" / "sigma_m_2472_71"

DATA_DIR_B = MAIN_DIR / "database" / "sigma_m_6593_89"
CONFIGS_FLUID = {
    "A": {
        "dual": Path(r"G:\BA Code\Evaluation\Data\config_r_max_7_5"),
        "eff": DATA_DIR_A / "eff"/ "23",
        "num": Path(r"G:\BA Code\Evaluation\Data\name2"),
        "sigma_0": 2472.71,
        "sigma_c":23
    },
    "B": {
        "dual": Path(r"G:\BA Code\Evaluation\Data\6config_r_max_7_5"),
        "eff": DATA_DIR_B / "eff"/ "62",
        "num": Path(r"G:\BA Code\Evaluation\Data\name"),  
     
    
        "7th":  DATA_DIR_B / "disp"/ "variation_n_shells" /"momente"/"moment_7",
        "3th":  DATA_DIR_B / "disp"/ "variation_n_shells" /"momente"/"moment_3",
        "sigma_0": 6593.89,
        "sigma_c":64
    },
}

LABEL = {"dual": "5th moment", "eff": "", "numeric": "", "3th": "3th moment", "7th": "7th moment", "rescaling":"rescaling"}

CONFIG_HALO = {
    "r_s": 1.28 * ut.kpc,
    "rho_s": 0.044174996 * ut.M_sun / (ut.pc**3),
    "w": 20.0, 
}


def Gravothermal_data_by_keys(keys: list, models: list) -> list:
    cfg = []

    RESCALE = False
    if "rescaling" in models:
        models = [m for m in models if m != "rescaling"]
        RESCALE = True

    for key in keys:
        for model in models:
            if not CONFIGS_FLUID[key].get(model):
                console.print(
                    f"[yellow]Skipping empty path for key: {key}, model: {model}[/yellow]"
                )
                continue

            console.print(
                f"[cyan]Processing Key: [bold]{key}[/bold] | Model: [bold]{model}[/bold][/cyan]"
            )
            sigma_0_units = CONFIGS_FLUID[key]["sigma_0"] * (ut.cm**2 / ut.g)

            dic = {
                "path": CONFIGS_FLUID[key][model],
                "run": str(key),
                "model": str(model),
                **sigma_eff(
                    CONFIG_HALO["r_s"],
                    CONFIG_HALO["rho_s"],
                    CONFIG_HALO["w"],
                    sigma_0_units,
                ),
                "sigma_0":sigma_0_units
            }
            print(dic)
            data_dict, _ = get_scaled_halo_dataset(dic["path"])
            data_dict = data_dict["dimensionful"]
            interp_keys = ["rho", "p"]
            t = data_dict["t"]
            r = data_dict["r"]

            func = {}
            for ik in interp_keys:
                func[f"f_{ik}"] = get_interpolation(t, r, data_dict[ik])

            rho_at_r = func["f_rho"](t, np.full_like(t, r_goal)).flatten()
            p_at_r = func["f_p"](t, np.full_like(t, r_goal)).flatten()

            nu= np.sqrt(p_at_r/rho_at_r)

            idx_min = np.argmin(rho_at_r)
            target_value = 666404109.645004 * eta
            idx_eta = np.argmin(np.abs(rho_at_r[idx_min:] - target_value)) + idx_min

            spline = CubicSpline(t, rho_at_r)

            d_spline = spline.derivative(1)
            roots_d = d_spline.roots()

            if len(roots_d) > 0:
                t_min_exact = roots_d[np.argmin(np.abs(roots_d - t[idx_min]))]
            else:
                t_min_exact = t[idx_min]

            rho_min_exact = float(spline(t_min_exact))

            spline_shifted = CubicSpline(t, rho_at_r - target_value)
            roots_target = spline_shifted.roots()

            valid_roots = roots_target[roots_target >= t_min_exact]

            if len(valid_roots) > 0:
                t_eta_exact = valid_roots[np.argmin(np.abs(valid_roots - t[idx_eta]))]
            else:
                t_eta_exact = t[idx_eta]

            rho_eta_exact = float(spline(t_eta_exact))

            extrema = {
                "tcf_idx": idx_min,
                "tcc_idx": idx_eta,
                "tcmin": (t_min_exact, rho_min_exact),
                "tcc": (t_eta_exact, rho_eta_exact),
            }

            cfg.append(
                {
                    **dic,
                    **func,
                    **extrema,
                    **data_dict,
                    "label": fr"{LABEL.get(model, '')} ({key})",
                    "style": model,
                    "nu": nu
                }
            )

    if RESCALE:
        for key in keys:
            dic= Gravothermal_data_by_keys((key),("eff", "eff"))[0]
            
            dic["t"] = new_t(CONFIGS_FLUID[key]["sigma_0"], CONFIGS_FLUID[key]["sigma_c"], dic["nu"], dic["t"])
            
            for ik in interp_keys:
                dic[f"f_{ik}"] = get_interpolation(dic["t"], dic["r"], dic[ik])
            dic["model"] = "rescaling" 
            dic["style"] = "rescaling"
            cfg.append(dic)   
    console.print(
        f"[green][bold]Success:[/bold] Processed {len(cfg)} configuration objects.[/green]"
    )
    return cfg

console.print("\n[bold magenta]=== Running Test ===[/bold magenta]")




def clean_val(v):
    """Truncate arrays longer than 3 elements for clean printing."""
    if isinstance(v, np.ndarray) and v.size > 3:
        return (
            f"[{', '.join(f'{x:.2f}' for x in v[:3])}, ... (shape: {v.shape})]"
        )
    return f"{v:.4f}" if isinstance(v, (float, np.float64)) else v




def base(path_fluid, model="dual", key="A"):
    data_dict, _ = get_scaled_halo_dataset(path_fluid)
    data_dict = data_dict["dimensionful"]
    interp_keys = ["rho", "p"]
    t = data_dict["t"]
    r = data_dict["r"]

    func = {}
    for ik in interp_keys:
        func[f"f_{ik}"] = get_interpolation(t, r, data_dict[ik])

    rho_at_r = func["f_rho"](t, np.full_like(t, r_goal)).flatten()
    p_at_r = func["f_p"](t, np.full_like(t, r_goal)).flatten()

    nu= np.sqrt(p_at_r/rho_at_r)

    idx_min = np.argmin(rho_at_r)
    target_value = 666404109.645004 * eta
    idx_eta = np.argmin(np.abs(rho_at_r[idx_min:] - target_value)) + idx_min

    spline = CubicSpline(t, rho_at_r)

    d_spline = spline.derivative(1)
    roots_d = d_spline.roots()

    if len(roots_d) > 0:
        t_min_exact = roots_d[np.argmin(np.abs(roots_d - t[idx_min]))]
    else:
        t_min_exact = t[idx_min]

    rho_min_exact = float(spline(t_min_exact))

    spline_shifted = CubicSpline(t, rho_at_r - target_value)
    roots_target = spline_shifted.roots()

    valid_roots = roots_target[roots_target >= t_min_exact]

    if len(valid_roots) > 0:
        t_eta_exact = valid_roots[np.argmin(np.abs(valid_roots - t[idx_eta]))]
    else:
        t_eta_exact = t[idx_eta]

    rho_eta_exact = float(spline(t_eta_exact))

    extrema = {
        "tcf_idx": idx_min,
        "tcc_idx": idx_eta,
        "tcmin": (t_min_exact, rho_min_exact),
        "tcc": (t_eta_exact, rho_eta_exact),
    }

    return {
            **func,
            **extrema,
            **data_dict,
            "label": fr"{LABEL.get(model, '')} ({key})",
            "style": model,
            "nu": nu,
            "run":key
        }
    
