import sys
from pathlib import Path
import numpy as np
from rich import print
import h5py
import astropy.units as u_astro
import astropy.constants as c_astro

def get_archive(path_archive):
    """
    Obtain dictionary of arrays of time-evolving halo quantities.
    """
    with h5py.File(path_archive, 'r') as hf:
        data = {key: hf[key][:] for key in hf.keys()}
    return data

def get_halo_initialization(path_ini):
    """
    Obtain saved halo initialization information and original halo state.
    """
    with h5py.File(path_ini, 'r') as hf:
        data1 = {}
        for key in hf.attrs.keys():
            if isinstance(hf.attrs[key], (bool, np.bool_)):
                data1[key] = bool(hf.attrs[key])
            else:
                data1[key] = hf.attrs[key]

        if len(hf.keys()) > 0:
            data2 = {key: hf[key][:] for key in hf.keys()}
        else:
            data2 = {}
    return data1, data2

def compute_physical_scales(init_data):
    """
    Computes conversion scales mapping dimensionless units to physical values
    using Astropy, replicating the core physical framework without full class overhead.
    """

    ut_kpc = u_astro.kpc
    ut_M_sun = u_astro.M_sun
    ut_pc = u_astro.pc
    ut_Gyr = u_astro.Gyr
    ct_G = c_astro.G

    r_s = init_data['r_s']
    rho_s = init_data['rho_s']

    scale_r = r_s * ut_kpc
    scale_rho = rho_s * ut_M_sun / ut_pc**3
    scale_m = 4.0 * np.pi * scale_rho * scale_r**3
    scale_u = ct_G * scale_m / scale_r
    scale_p = scale_u * scale_rho
    scale_t = 1.0 / np.sqrt(4.0 * np.pi * scale_rho * ct_G)

    return {
        'r': scale_r,
        'rho': scale_rho,
        'm': scale_m,
        'p': scale_p,
        't': scale_t
    }

def get_pure_archive_data(run_path, inspect=False):
    """
    Loads dimensionless data directly from the archive and initialization scaling 
    parameters from halo_ini.h5, bypassing any evolution logic.
    """
    run_dir = Path(run_path)
    folder_name = run_dir.resolve().name
    archive_file_path = run_dir / f"{folder_name}.h5" 
    
    if not archive_file_path.exists():
        h5_files = [f for f in run_dir.glob("*.h5") if f.name != "halo_ini.h5"]
        if h5_files:
            archive_file_path = h5_files[0]
        else:
            raise FileNotFoundError(f"No suitable .h5 archive file found in {run_dir}")

    data_dimless = get_archive(str(archive_file_path))
    
    try:
        initialization_data = get_halo_initialization(run_dir / 'halo_ini.h5')[0]
    except Exception as e:
        raise FileNotFoundError(f"Failed to load vital initialization metadata: {e}")

    if inspect:
        print(f"\n[bold cyan]--- Pure Archive ({archive_file_path.name}) & Initialization Inspection ---[/bold cyan]")
        print(f"[bold]Available keys in the dimensionless archive (Time Evolution):[/bold]")
        print(f"  {list(data_dimless.keys())}")
        print(f"\n[bold]Available keys in initialization (Physical Scaling):[/bold]")
        print(f"  {list(initialization_data.keys())}")
        print("[bold cyan]--------------------------------------------------\n[/bold cyan]")

    return data_dimless, initialization_data

def get_scaled_halo_dataset(run_path, target_units=None, inspect=False):
    """
    Main API function. Extracts and converts dimensionless archive parameters into 
    physical quantities based on computed scales, alongside documented initial conditions.
    
    Parameters
    ----------
    run_path : str or Path
        Directory containing the simulation outputs.
    target_units : dict, optional
        Custom Astropy units for conversion output. Defaults to standard astrophysical units.
    inspect : bool, optional
        If True, outputs directory analysis structural data.
        
    Returns
    -------
    data_dict : dict
        Contains 'dimensionless' and 'dimensionful' sub-dictionaries of primary arrays.
    meta_dict : dict
        The exact configuration parameters with human-readable string units.
    """
    default_units = {
        'r': u_astro.kpc,
        'rho': u_astro.M_sun / u_astro.kpc**3,
        'm': u_astro.M_sun,
        'p': u_astro.M_sun / (u_astro.kpc * u_astro.Gyr**2),
        't': u_astro.Gyr
    }
    if target_units is not None:
        default_units.update(target_units)

    data_dimless, initialization_data = get_pure_archive_data(run_path, inspect=inspect)
    
    scales = compute_physical_scales(initialization_data)

    data_dict = {
        'dimensionless': {},
        'dimensionful': {}
    }
    
    keys_to_scale = ['r', 'rho', 'm', 'p', 't']
    for key in keys_to_scale:
        if key in data_dimless:
            raw_array = data_dimless[key]
            data_dict['dimensionless'][key] = raw_array
            
            scale_factor = scales[key]
            target_unit = default_units[key]
            data_dict['dimensionful'][key] = (raw_array * scale_factor).to_value(target_unit)

    meta_dict = {}
    unit_annotations = {
        'r_s': 'kpc',
        'rho_s': 'M_sun/pc^3',
        'sigma_m_with_units': 'cm^2/g',
        'w_units': 'km/s'
    }
    
    for key, val in initialization_data.items():
        if key in unit_annotations:
            meta_dict[key] = f"{val} [{unit_annotations[key]}]"
        else:
            meta_dict[key] = val
    print(meta_dict)
    return data_dict, meta_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extracts dimless/dimensionful datasets and initial conditions cleanly from a halo run."
    )
    parser.add_argument("run_path", type=str, help="Path to the simulation data directory")
    parser.add_argument("--inspect", action="store_true", help="Activates structural key inspection mode")
    args = parser.parse_args()

    cleaned_path = args.run_path.strip().replace('\xa0', ' ')
    print(f"\n[bold green]Processing run directory directly:[/bold green] {cleaned_path}")

    try:
        halo_data, initial_conditions = get_scaled_halo_dataset(cleaned_path, inspect=args.inspect)
        
        print("[bold green]Success![/bold green] Dimensionful arrays and metadata cleanly isolated.")
        print(f"\n[bold cyan]Dimensionful Keys Extracted:[/bold cyan] {list(halo_data['dimensionful'].keys())}")
        print(f"[bold cyan]Total Config Metadata Keys Extracted:[/bold cyan] {len(initial_conditions)}")
        
        if 't' in halo_data['dimensionful']:
            print(f"[green]Verification - Extracted physical times vector (Gyr):[/green] {halo_data['dimensionful']['t'][:3]}...")

    except Exception as e:
        print(f"[bold red]Pipeline execution failed:[/bold red] {e}")