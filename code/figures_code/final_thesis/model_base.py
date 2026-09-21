from reader_code.Nbody_reader_code.read_data_fischer import get_fischer_dataset
from data_path import gravothermal_root, Nbody_root

SIMULATION_CONFIGS = {
    "B":
    {"name": "Fischer_R", 
     "label": "B", 
     "model": "B", 
     "target_base_val": 666404109.645004,
     "sigma_eff": 64,
     "range_min":[0.2,2], 
     "range_thresh":[4,8],
     "sigma_0":6593.89,
     }
}

CONFIGS_FLUID = {
    "A":{
        "dual":     r"D:\BA Code\Evaluation\Data\Halo1_2472d71_rmin0d0078125",
        "eff":      r"D:\BA Code\Evaluation\Data\Halo1_sig23",
        "nummeric": r"D:\BA Code\Evaluation\Data\name2",
        "eff":      23,
        "sigma_0":  2472.71
        },
    "B":{
        "dual":     r"D:\BA Code\Evaluation\Data\Halo1_sig6593d89_rmin00078125_rmax7d8125Old",
        "eff":      r"D:\BA Code\Evaluation\Data\Halo1_sig64",
        "nummeric": r"D:\BA Code\Evaluation\Data\name",
        "eff":      64,
        "sigma_0":  6593.89
        },
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
        
        nbody_data.append({
            "model": cfg["run"],
            "label": cfg["label"],
            "rho_2d": rho_2d,          
            "rho_err_2d": rho_err_2d,  
            "r": r_arr,       
            "t": t_arr            
        })
    return nbody_data