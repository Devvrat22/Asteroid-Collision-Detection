

import numpy as np
import pandas as pd

def kepler_solver(M, e, tol=1e-10, max_iter=100):
     """
    Solve Kepler's equation:
        M = E - e*sin(E)
    for eccentric anomaly E.

    Parameters:
        M : float
            Mean anomaly (radians)
        e : float
            Eccentricity
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations

    Returns:
        E : float
            Eccentric anomaly (radians)
    """
      # Initial guess
     E = M if e < 0.8 else np.pi

     for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)

        delta = f / f_prime
        E -= delta

        if abs(delta) < tol:
            break

     return E