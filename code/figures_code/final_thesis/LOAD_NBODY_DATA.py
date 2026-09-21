from reader_code.Nbody_reader_code.read_data_fischer import get_fischer_dataset
from analyse_code.Nbody_prep import analyze_and_fit_data, avrg_density
from data_path import gravothermal_root, Nbody_root
import numpy as np

SIMULATION_CONFIGS = {
    "A":{"name": "Fischer_A", "run":"A", "label": "A", "sigma_0": 2472.71, "c": 0.0, "eff":23, "model":"A","range_min":[0.2,2], "range_thresh":[10,18], "target_base_val":666404109.645004},
    "B":
    {"name": "Fischer_R", 
     "label": "B", 
     "run": "B", 
     "target_base_val": 666404109.645004,
     "sigma_eff": 64,
     "range_min":[0.2,2], 
     "range_thresh":[6.631,7 ],
     "sigma_0":6593.89,
     },
    "C":
    {"name": "Fischer_Q", 
     "label": "C", 
     "run": "C", 
     "target_base_val": 666404109.645004,
     "sigma_eff": None,
     "range_min":[0.2,2], 
     "range_thresh":[3,8],
     "sigma_0":None,
     }
}


CONFIG_HALO = {
    "r_s" : 2.783e-1,
    "rho_s": 4.417e-3
}

def get_configs_Nbody_by_keys(keys: list) -> list:
    """
    Returns a list (array) of dictionaries based on a list of provided keys 
    (e.g., ['B'] or ['A', 'C']).
    """
    return [SIMULATION_CONFIGS[key] for key in keys if key in SIMULATION_CONFIGS]


def Nbody_data_by_keys(keys: list) -> list:
    config = get_configs_Nbody_by_keys(keys)
    nbody_data = []
    
    for cfg in config:
        data_dict, _ = get_fischer_dataset(Nbody_root(cfg["name"]))
        dim_data = data_dict["dimensionful"]
        
        rho_2d = dim_data["rho"][:, :]
        rho_err_2d = dim_data["rho_err"][:, :]
        r_arr = dim_data["r_mid"][:]
        t_arr = dim_data["t"][:]

        N_START = 8
        N_END = 12

        rho_avrg, rho_erro_avrg = [], []
        for d, err in zip(rho_2d[:, N_START:N_END], rho_err_2d[:, N_START:N_END]):

            rh, e, r = avrg_density(d, err, r_arr[N_START:N_END])
            rho_avrg.append(rh)
            rho_erro_avrg.append(e)
            r_goal_extracted = r  
        t_exact_min, rho_exact_min, t_exact_thresh, rho_exact_thresh = analyze_and_fit_data(
            t_arr, np.array(rho_avrg), 3, cfg["range_min"], cfg["range_thresh"], cfg["target_base_val"]
        )

        nbody_data.append({
            "run": cfg["run"],
            "label": cfg["label"],
            "rho_2d": rho_2d,          
            "rho_err_2d": rho_err_2d,  
            "r": r_arr,             
            "t": t_arr,
            "r_goal": r_goal_extracted,
            "tcmin": (t_exact_min, rho_exact_min),
            "tcc": (t_exact_thresh, rho_exact_thresh), 
            "rho_avrg": np.array(rho_avrg),
            "rho_avrg_err": np.array(rho_erro_avrg),
            "style": "nbody_benchmark",
            "tcf_idx":np.argmin(abs(t_arr-t_exact_min)),
            **dim_data
        })

    return nbody_data