import logging
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


INTEGRATION_DEFAULTS = {
    "epsabs": 1e-40,
    "epsrel": 1e-8,
    "limit": 500,
    "upper_lim": np.inf,
    "tolerance": 0.1,
    "brake_at_error": False,
    "verbose": True,
}


def denominator(nu, n):
    return 0.5 * (2.0 * nu) ** (n + 1.0) * gamma((n + 1.0) / 2.0)


def define_integrand(shape_func, nu, n):
    return lambda x: (x ** n) * np.exp(-(x ** 2) / (4.0 * nu ** 2)) * shape_func(x)


def perform_integration(integrand, **kwargs):
    epsabs = kwargs.get("epsabs", INTEGRATION_DEFAULTS["epsabs"])
    epsrel = kwargs.get("epsrel", INTEGRATION_DEFAULTS["epsrel"])
    limit = kwargs.get("limit", INTEGRATION_DEFAULTS["limit"])
    upper_lim = kwargs.get("upper_lim", INTEGRATION_DEFAULTS["upper_lim"])
    tolerance = kwargs.get("tolerance", INTEGRATION_DEFAULTS["tolerance"])
    brake_at_error = kwargs.get("brake_at_error", INTEGRATION_DEFAULTS["brake_at_error"])
    verbose = kwargs.get("verbose", INTEGRATION_DEFAULTS["verbose"])

    try:
        result, error = quad(
            integrand,
            0.0,
            upper_lim,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
    except Exception as exc:
        if verbose:
            print(f"Integration failed: {exc}")
        return np.nan

    error_percentage = (error / abs(result)) * 100.0 if result != 0 else np.inf

    if error_percentage > tolerance:
        msg = (
            f"Integration error: estimate {error:.4e} "
            f"equals {error_percentage:.2f}% and exceeds {tolerance}%"
        )

        if brake_at_error:
            raise ValueError(msg)

        logging.warning(msg)

    return result


def averaged_viscosity_shape(shape_func, nu, n, **kwargs):
    numerator = perform_integration(
        define_integrand(shape_func, nu, n),
        **kwargs,
    )

    return numerator / denominator(nu, n)


def compute_moment_array(shape_func, nu_values, n, sigma_0=1.0, **kwargs):
    values = []

    for nu in nu_values:
        value = averaged_viscosity_shape(
            shape_func=shape_func,
            nu=nu,
            n=n,
            **kwargs,
        )

        values.append(sigma_0 * value)

    return np.asarray(values)


def save_csv(nu_values, y_values, filename, x_name="x", y_name="sigma_vis"):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    data = np.column_stack([nu_values, y_values])
    header = f"{x_name},{y_name}"

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=header,
        comments="",
    )

    print(f"Saved: {filename}")


def moment_to_name(n):
    return str(n).replace(".", "p")


def run_viscosity_moment_grid(
    shape_func,
    moments,
    nu_values,
    output_dir,
    sigma_0=1.0,
    x_name="x",
    y_prefix="sigma_vis_moment",
    **kwargs,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for n in moments:
        print(f"Berechne Moment n = {n}")

        y_values = compute_moment_array(
            shape_func=shape_func,
            nu_values=nu_values,
            n=n,
            sigma_0=sigma_0,
            **kwargs,
        )

        n_name = moment_to_name(n)
        filename = output_dir / f"{y_prefix}_{n_name}.csv"

        save_csv(
            nu_values=nu_values,
            y_values=y_values,
            filename=filename,
            x_name=x_name,
            y_name=f"{y_prefix}_{n_name}",
        )

        results[n] = y_values

    return results