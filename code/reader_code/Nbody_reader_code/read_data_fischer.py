import os
import numpy as np
from pathlib import Path
from rich import print
from reader_code.Nbody_reader_code.io_Fischer import read_dict

FISCHER_UNITS = {
    'MU_to_Msun': 1e10,      
    't_to_Gyr': 0.979,        
    'r_to_kpc': 1.0,         
    'rho_to_Msun_kpc3': 1e10, 
    'v_to_kms': 1.0 / 0.979  
}

def get_fischer_dataset(run_path, inspect=False):
    """
    Extracts and converts dimensionless profile datasets and errors from the Fischer 
    framework into physical quantities, mirroring a standardized data pipeline structure.

    Parameters
    ----------
    run_path : str or Path
        Directory containing the simulation outputs (specifically 'results_profiles.txt').
    inspect : bool, optional
        If True, outputs structural key inspection and validation data.

    Returns
    -------
    data_dict : dict
        Contains 'dimensionless' and 'dimensionful' sub-dictionaries of primary profiles.
    meta_dict : dict
        Metadata configuration parameters extracted from the header info.
    """
    run_dir = Path(run_path)
    profile_file_path = run_dir / "results_profiles.txt"

    if not profile_file_path.exists():
        raise FileNotFoundError(f"Fischer profile file not found at: {profile_file_path}")

    raw_data = read_dict(str(profile_file_path))
    
    time_steps = raw_data["Time"]
    profile_edges = raw_data["ProfileEdges"]
    mid_radii = (profile_edges[:-1] + profile_edges[1:]) / 2.0

    num_snaps = len(time_steps)
    num_bins = len(mid_radii)


    raw_mass = raw_data["ProfileMassDM"]
    raw_mass_err = raw_data["ProfileMassErrDM"]
    raw_central_mass = raw_data["CentralMassDM"]
    raw_central_mass_err = raw_data["CentralMassErrDM"]
    
    raw_v2 = raw_data["ProfileVelDisprRadDM"]
    raw_v = np.sqrt(np.maximum(raw_v2, 0.0)) 
    
    raw_v2_err = raw_data.get("ProfileVelDisprRadErrDM", np.zeros_like(raw_v2))
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_v_err = raw_v2_err / (2.0 * raw_v)
    raw_v_err = np.nan_to_num(raw_v_err, nan=0.0, posinf=0.0, neginf=0.0)

    raw_enclosed_mass = np.zeros((num_snaps, num_bins))
    raw_enclosed_mass_err = np.zeros((num_snaps, num_bins))

    for snap in range(num_snaps):
        for bin_idx in range(num_bins):
            raw_enclosed_mass[snap, bin_idx] = (
                raw_central_mass[snap] + np.sum(raw_mass[snap][:bin_idx ])
            )
            raw_enclosed_mass_err[snap, bin_idx] = (
                raw_central_mass_err[snap] + np.sum(raw_mass_err[snap][:bin_idx ])
            )

    data_dict = {
        'dimensionless': {
            't': time_steps,
            'r_edges': profile_edges,
            'r_mid': mid_radii,
            'rho': raw_data["ProfileDensityDM"],
            'rho_err': raw_data["ProfileDensityErrDM"],
            'm_shell': raw_mass,
            'm_shell_err': raw_mass_err,
            'm_enclosed': raw_enclosed_mass,
            'm_enclosed_err': raw_enclosed_mass_err,
            'v': raw_v,         
            'v_err': raw_v_err,  
            'v_t': raw_data["ProfileVelDisprTangDM"]**(1/2),
            'v_t_err': raw_data["ProfileVelDisprTangErrDM"]**(1/2),
            'beta': np.array(1-raw_data["ProfileVelDisprTangDM"]/ raw_data["ProfileVelDisprRadDM"])

        },
        'dimensionful': {}
    }

    dimless = data_dict['dimensionless']
    

    data_dict['dimensionful']['t'] = dimless['t'] * FISCHER_UNITS['t_to_Gyr']
    data_dict['dimensionful']['r_edges'] = dimless['r_edges'] * FISCHER_UNITS['r_to_kpc']
    data_dict['dimensionful']['r_mid'] = dimless['r_mid'] * FISCHER_UNITS['r_to_kpc']

    data_dict['dimensionful']['rho'] = dimless['rho'] * FISCHER_UNITS['rho_to_Msun_kpc3']
    data_dict['dimensionful']['rho_err'] = dimless['rho_err'] * FISCHER_UNITS['rho_to_Msun_kpc3']

    data_dict['dimensionful']['m_shell'] = dimless['m_shell'] * FISCHER_UNITS['MU_to_Msun']
    data_dict['dimensionful']['m_shell_err'] = dimless['m_shell_err'] * FISCHER_UNITS['MU_to_Msun']
    data_dict['dimensionful']['m_enclosed'] = dimless['m_enclosed'] * FISCHER_UNITS['MU_to_Msun']
    data_dict['dimensionful']['m_enclosed_err'] = dimless['m_enclosed_err'] * FISCHER_UNITS['MU_to_Msun']

    data_dict['dimensionful']['v'] = dimless['v'] * FISCHER_UNITS['v_to_kms']
    data_dict['dimensionful']['v_err'] = dimless['v_err'] * FISCHER_UNITS['v_to_kms']
    data_dict['dimensionful']['v_t'] = dimless['v_t'] * FISCHER_UNITS['v_to_kms']
    data_dict['dimensionful']['beta'] = dimless['beta']

    meta_dict = {
        "Info": raw_data.get("Info", "No metadata string provided."),
        "unit_metadata": {
            "t": "Gyr",
            "r": "kpc",
            "rho": "M_sun/kpc^3",
            "m": "M_sun",
            "v": "km/s"
        }
    }

    if inspect:
        print(f"\n[bold cyan]--- Fischer Pipeline Archive Conversion Inspection ({profile_file_path.name}) ---[/bold cyan]")
        print(f"[bold]Available Keys under 'dimensionful':[/bold]")
        print(f"  {list(data_dict['dimensionful'].keys())}")
        print(f"\n[bold]Metadata Info String Extracted:[/bold]")
        print(f"  {meta_dict['Info']}")
        print("[bold cyan]--------------------------------------------------------------------------\n[/bold cyan]")

    return data_dict, meta_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Standardized pipeline to extract dimensionful/dimensionless data stacks from Fischer profiles."
    )
    parser.add_argument("run_path", type=str, help="Path to directory containing results_profiles.txt")
    parser.add_argument("--inspect", action="store_true", help="Activates clear validation prints")
    args = parser.parse_args()

    cleaned_path = args.run_path.strip().replace('\xa0', ' ')
    print(f"\n[bold green]Initializing Fischer extraction pipeline for path:[/bold green] {cleaned_path}")

    try:
        halo_data, metadata = get_fischer_dataset(cleaned_path, inspect=args.inspect)
        print("[bold green]Success![/bold green] Fischer profiles mapped smoothly to physical unit dimensions.")
        

        if 'm_enclosed' in halo_data['dimensionful']:
            sample_mass = halo_data['dimensionful']['m_enclosed'][0][:3]
            sample_err = halo_data['dimensionful']['m_enclosed_err'][0][:3]
            print(f"[green]Validation - Initial Enclosed Masses (M_sun):[/green] {sample_mass}")
            print(f"[green]Validation - Initial Enclosed Mass Errors (M_sun):[/green] {sample_err}")
            

        if 'v' in halo_data['dimensionful']:
            sample_v = halo_data['dimensionful']['v'][0][:3]
            sample_v_err = halo_data['dimensionful']['v_err'][0][:3]
            print(f"[green]Validation - Initial Velocities (km/s):[/green] {sample_v}")
            print(f"[green]Validation - Initial Velocity Errors (km/s):[/green] {sample_v_err}")

    except Exception as e:
        print(f"[bold red]Pipeline execution failed:[/bold red] {e}")