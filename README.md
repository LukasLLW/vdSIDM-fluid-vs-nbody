# On the Applicability of Gravothermal Fluid Approaches for Velocity Dependent SIDM: A Direct Comparison with N-body Simulations

This repository serves as a summary of the most important code elements from my bachelor's thesis.

---

## Directory Overview

### [`code/`](./code)
Contains all executable routines and module definitions for running simulations and analyzing data.

* **[`code/GravothermalSIDM`](./code/GravothermalSIDM)**: Contains a modification file with custom changes to be applied to the original [GravothermalSIDM](https://github.com/kboddy/GravothermalSIDM) source code.
* **[`code/reader_code/`](./code/reader_code)**: Adapted modules for reading data outputs from Fischer models and GravothermalSIDM.
* **[`code/analyse_code/`](./code/analyse_code)**: Processing algorithms described in the thesis for advanced data analysis beyond basic reading and plotting.
* **[`code/figures_code/`](./code/figures_code)**: Plotting scripts designed to generate thesis-ready figures ([`final_thesis/`](./code/figures_code/final_thesis)).

### [`database/`](./database)

*(Note: Simulation data and datasets are not tracked in this repository due to storage constraints. If you require access to the raw data or datasets to reproduce or extend the results, please feel free to reach out to the repository owner via email.)*

When provided, the database structure contains data for the following scenarios:

* **Simulation Runs by Cross-Section (**$\sigma/m$**):**
  * **`database/sigma_m_2472_71/`**: Dataset for $\sigma/m = 2472.71$ runs.
  * **`database/sigma_m_6593_89/`**: Dataset for $\sigma/m = 6593.89$ runs.
* **Structure within Simulation Directories:**
  * **`disp/`**: Analytic velocity-dependent cross-section runs using velocity moments ($3^{\text{rd}}$, $5^{\text{th}}$, and $7^{\text{th}}$ moments).
  * **`num/`**: Numerical velocity-dependent cross-section runs using velocity moments ($3^{\text{rd}}$, $5^{\text{th}}$, and $7^{\text{th}}$ moments).
  * **`eff/`**: Effective cross-section runs.
* **Parameter Variations:**
  * `variation_n_shells`: Spatial grid resolution tests (50 to 250 shells).
  * `variation_r_min` / `variation_r_max`: Inner boundary and spatial extent verification.
* **`database/Fischer_Data/`**: External reference datasets (`Fischer_A/`, `Fischer_Q/`, `Fischer_R/`). These reference datasets can also be accessed online at the [USM Data Repository](https://homepages.usm.uni-muenchen.de/mfischer/data/DSMC/).
* **`database/viscosity_moment_csv/`**: Exported numerical data tables for velocity moments ($3^{\text{rd}}$, $5^{\text{th}}$, and $7^{\text{th}}$ moments).


## Setup & Execution

Before running any script from the `code/` directory, ensure that the root directory is set in your `PYTHONPATH` environment variable:

* **Linux / macOS:**
  ```bash
  export PYTHONPATH="."
  ```
* **Windows (PowerShell):**
  ```powershell
  $env:PYTHONPATH="."
  ```

Additionally, set up the required directory structure before running simulations:

1. Clone or download the [GravothermalSIDM](https://github.com/kboddy/GravothermalSIDM) repository into `./code/simulation_code/SourcePy/` and copy the modification file into that directory.
2. Create a `database/` directory in the root folder to store output and simulation data.

* **Linux / macOS:**
  ```bash
  mkdir -p code/simulation_code/SourcePy database
  ```
* **Windows (PowerShell):**
  ```powershell
  New-Item -ItemType Directory -Force -Path "code/simulation_code/SourcePy", "database"
  ```