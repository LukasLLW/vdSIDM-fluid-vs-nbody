import numpy as np
from pathlib import Path

from analyse_code.thermal_viscosity_moments import run_viscosity_moment_grid


OUTPUT_DIR = Path("viscosity_moment_csv")

SIGMA_0 = 1.0

NU_VALUES = np.logspace(-3, 3, 600)

INTERNAL_MOMENTS = np.linspace(6, 8, 10)
OUTPUT_MOMENTS = INTERNAL_MOMENTS - 2


def sigma_viscosity_shape(x):
    return 1.0 / (1.0 + x ** 2) ** 2


if __name__ == "__main__":
    for internal_n, output_n in zip(INTERNAL_MOMENTS, OUTPUT_MOMENTS):
        run_viscosity_moment_grid(
            shape_func=sigma_viscosity_shape,
            moments=[internal_n],
            nu_values=NU_VALUES,
            output_dir=OUTPUT_DIR,
            sigma_0=SIGMA_0,
            x_name="v_over_w",
            y_prefix=f"sigma_vis_{str(output_n).replace('.', 'p')}",
            epsabs=1e-40,
            epsrel=1e-8,
            limit=500,
            upper_lim=np.inf,
            tolerance=0.1,
            brake_at_error=False,
            verbose=True,
        )