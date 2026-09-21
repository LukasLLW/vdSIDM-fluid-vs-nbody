
from scipy.special import exp1
from scipy.integrate import quad
import numpy as np
def visco(x):
    """Base viscosity function."""
    return (1 + x**2)**(-2)

def moments(model_name):
    """Analytic moment models (Fischer models)."""
    if "Fischer" in model_name:
        phi = lambda x: 4.0 * x ** 2

        def fischer_function_3(v):
            p = phi(v)
            if p < 1e-10:
                return 1.0
            term1 = 1.0 + 1.0 / p
            term2 = (2.0 / p + 1.0 / p ** 2) * np.exp(1.0 / p) * exp1(1.0 / p)
            return 1.0 / (2.0 * p ** 2) * (term1 - term2)

        def fischer_function_5(v):
            p = phi(v)
            if p < 1e-10:
                return 1.0
            term1 = p - 2.0 - 1.0 / p
            term2 = (3.0 / p + 1.0 / p ** 2) * np.exp(1.0 / p) * exp1(1.0 / p)
            return 1.0 / (6.0 * p ** 3) * (term1 + term2)

        def fischer_function_7(v):
            phi_val = 4.0 * v ** 2
            if phi_val < 0.0005:
                return 1.0 - 5.0 * phi_val + 15.0 * phi_val ** 2
            if phi_val < 50.0:
                iterand = lambda x: (x ** 4 * np.exp(-x)) / (1.0 + phi_val * x) ** 2
                res, _ = quad(iterand, 0.0, np.inf)
                return res / 24.0
            return 0.0

        nu_interp = np.logspace(-5, 3, 8000)
        model_lower = model_name.lower()

        if "7" in model_lower:
            y = np.array([fischer_function_7(x) for x in nu_interp])
        elif "5" in model_lower:
            y = np.array([fischer_function_5(x) for x in nu_interp])
        elif "3" in model_lower:
            y = np.array([fischer_function_3(x) for x in nu_interp])
        elif "viscosity" in model_lower:
            y = 1.0 / ((1.0 + nu_interp ** 2) ** 2)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return lambda x: np.interp(x, nu_interp, y)

    raise ValueError(f"Unknown model: {model_name}")
