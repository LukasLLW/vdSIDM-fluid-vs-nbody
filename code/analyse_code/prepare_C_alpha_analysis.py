import os
import re
from rich.console import Console
from rich.table import Table
from analyse_code.Gravothermal_interpolation import get_interpolation
from reader_code.gravothermal_reader_code.read_Nishikawa_Data import get_scaled_halo_dataset
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import matplotlib.pyplot as plt
console = Console()

def load_Fluid_Grid(base_path, overview=True):
    """
    Scans the base directory for fluid configuration folders,
    extracts C and alpha values using regex, and returns a list of configs.
    """
    if overview:
        console.print("[bold cyan]LAUNCHING[/bold cyan] Loading C-Alpha-grid started...")

    pattern_alpha = re.compile(r"alpha_([0-9.]+)", re.IGNORECASE)
    pattern_C = re.compile(r"C_([0-9.]+)", re.IGNORECASE)

    fluid_configs = []
    unique_c = set()
    unique_alpha = set()
    existing_combinations = set()

    with os.scandir(base_path) as entries:
        for entry in entries:
            if entry.is_dir():
                match_alpha = pattern_alpha.search(entry.name)
                match_C = pattern_C.search(entry.name)
                
                if match_alpha and match_C:
                    alpha_str = match_alpha.group(1).rstrip(".")
                    c_str = match_C.group(1).rstrip(".")
                    
                    alpha_val = float(alpha_str) if "." in alpha_str else int(alpha_str)
                    c_val = float(c_str) if "." in c_str else int(c_str)
                    
                    data_dic,_ =get_scaled_halo_dataset(entry.path)
                    data_dic = data_dic["dimensionful"]
                    rho = data_dic["rho"]
                    t = data_dic["t"]
                    r = data_dic["r"]
                    fluid_configs.append({
                        "path": entry.path,
                        "C": c_val,
                        "rho": rho, "r":r, "t":t,
                        "alpha": alpha_val
                    })
                    
                    unique_c.add(c_val)
                    unique_alpha.add(alpha_val)
                    existing_combinations.add((c_val, alpha_val))

    if overview:
        console.print(fr"[magenta]LOADING[/magenta]   {len(fluid_configs)} runs loaded from [yellow]{base_path}[/yellow]...\n")
        
        if fluid_configs:
            sorted_c = sorted(list(unique_c))
            sorted_alpha = sorted(list(unique_alpha))

            
            table = Table(title="DETECTED C-ALPHA GRID MATRIX", title_style="bold magenta")
            
            table.add_column("C / Alpha", style="cyan", justify="right")
            for alpha in sorted_alpha:
                table.add_column(str(alpha), justify="center", style="yellow")
            
            for c in sorted_c:
                row_cells = [str(c)]
                for alpha in sorted_alpha:
                    if (c, alpha) in existing_combinations:
                        row_cells.append("[bold green][X][/bold green]")
                    else:
                        row_cells.append("[red]-[/red]")
                table.add_row(*row_cells)
            
            console.print(table)
            console.print(f"[bold green]SUCCESS[/bold green] Grid ingestion completed.\n")

    return fluid_configs

import numpy as np
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator

def build_4d_interpolator(fluid_config, overview=True):
    if overview:
        console.print("[bold cyan]LAUNCHING[/bold cyan] Constructing 4D continuous interpolator...")

    fluid_data = []
    for cfg in fluid_config:
        data_dict, _ = get_scaled_halo_dataset(cfg["path"])
        fluid_data.append({
            "r": np.array(data_dict["dimensionful"]["r"]),
            "rho": data_dict["dimensionful"]["rho"],
            "t": data_dict["dimensionful"]["t"],
            "f_interp": get_interpolation(data_dict["dimensionful"]["t"], data_dict["dimensionful"]["r"], data_dict["dimensionful"]["rho"]),
            "C": cfg["C"],
            "alpha": cfg["alpha"]
        })

    rmax = min(np.min(cfg["r"][:,-1]) for cfg in fluid_data)
    rmin = max(np.max(cfg["r"][:,0]) for cfg in fluid_data)
    tmax = max(np.min(cfg["t"][-1] for cfg in fluid_data))
    tmin = min(np.min(cfg["t"]) for cfg in fluid_data)
    print(rmax, rmin, tmax, tmin)
    r_targets = np.logspace(np.log10(rmin), np.log10(rmax), 100)
    t_targets = np.linspace(tmin, tmax, 400)
    

    if overview:
        console.print(f"[magenta]LOADING[/magenta]   Target grid dimensions generated: r={len(r_targets)}, t={len(t_targets)}")
    
    T_targets_2d, R_targets_2d = np.meshgrid(t_targets, r_targets, indexing='ij')
    t_flat = T_targets_2d.ravel()
    r_flat = R_targets_2d.ravel()
    df = pd.DataFrame(fluid_config)

    unique_c = np.sort(df["C"].unique())
    unique_alpha = np.sort(df["alpha"].unique())

    shape = (len(unique_c), len(unique_alpha), len(t_targets), len(r_targets))
    nan_grid_4d = np.full(shape, np.nan)

    for idx_c, CU in enumerate(unique_c):
        df_special = df[df["C"] == CU]

        f_2d_grids = [
            row["f_rho"](t_flat, r_flat).reshape(len(t_targets), len(r_targets))
            for _, row in df_special.iterrows()
        ]
        alpha_concret = [
            row["alpha"]
            for _, row in df_special.iterrows()
        ]
        interp_3d = RegularGridInterpolator((alpha_concret, t_targets, r_targets), np.stack(f_2d_grids, axis=0), bounds_error=False, fill_value=None, method="linear")

        ALPHA_grid, T_grid, R_grid = np.meshgrid(unique_alpha, t_targets, r_targets, indexing='ij')
        eval_points = np.column_stack((ALPHA_grid.ravel(), T_grid.ravel(), R_grid.ravel()))
        final_3d_grid = interp_3d(eval_points).reshape(len(unique_alpha), len(t_targets), len(r_targets)) 
        
        nan_grid_4d[idx_c, :, :, :] = final_3d_grid
    interp_4d = RegularGridInterpolator((unique_c, unique_alpha, t_targets, r_targets), nan_grid_4d, bounds_error=False, fill_value=None)
    return interp_4d, unique_c, unique_alpha

def evaluate_interpolator_slice(interpolator, r_input, t_input, c_input, alpha_input, overview=True):
    if overview:
        console.print("[bold cyan]LAUNCHING[/bold cyan] Evaluating 4D interpolator slice...")

    r_arr = np.atleast_1d(r_input)
    t_arr = np.atleast_1d(t_input)
    c_arr = np.atleast_1d(c_input)
    alpha_arr = np.atleast_1d(alpha_input)

    if overview:
        console.print(
            f"[magenta]LOADING[/magenta]   Input shapes: "
            f"r={r_arr.shape}, t={t_arr.shape}, C={c_arr.shape}, Alpha={alpha_arr.shape}"
        )

    R, T, C, A = np.meshgrid(r_arr, t_arr, c_arr, alpha_arr, indexing='ij')

    query_points = np.column_stack((R.ravel(), T.ravel(), C.ravel(), A.ravel()))

    flat_results = interpolator(query_points)

    output_shape = (len(r_arr), len(t_arr), len(c_arr), len(alpha_arr))
    result_grid = flat_results.reshape(output_shape)

    final_output = np.squeeze(result_grid)

    if overview:
        console.print(f"[bold green]SUCCESS[/bold green] Slice evaluation completed. Output shape: {final_output.shape}\n")

    return final_output

if __name__ == "__main__":
    database_path = r"C:\Users\lukas\Documents\GitHub_new26\DATABASE\BACHELOR\H5\R_alpha_C_grid_moment_7"
    
    fluid_configs = load_Fluid_Grid(base_path=database_path)
    INTERP, C, ALPHA = build_4d_interpolator(fluid_config=fluid_configs)
    
    t_samples = np.linspace(0, 7, 50)
    
    density_slice = evaluate_interpolator_slice(
        interpolator=INTERP,
        r_input=0.033,
        t_input=t_samples,
        c_input=C,
        alpha_input=1.2,
        overview=True
    )