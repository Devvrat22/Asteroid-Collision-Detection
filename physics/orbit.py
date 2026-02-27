import numpy as np
from physics.kepler_solver import kepler_solver


def rotation_matrix(omega_big, i, omega_small):
    """
    R = Rz(Ω) * Rx(i) * Rz(ω)
    """
    cos_O = np.cos(omega_big)
    sin_O = np.sin(omega_big)

    cos_i = np.cos(i)
    sin_i = np.sin(i)

    cos_w = np.cos(omega_small)
    sin_w = np.sin(omega_small)

    Rz_O = np.array([
        [cos_O, -sin_O, 0],
        [sin_O,  cos_O, 0],
        [0, 0, 1]
    ])

    Rx_i = np.array([
        [1, 0, 0],
        [0, cos_i, -sin_i],
        [0, sin_i, cos_i]
    ])

    Rz_w = np.array([
        [cos_w, -sin_w, 0],
        [sin_w,  cos_w, 0],
        [0, 0, 1]
    ])

    return Rz_O @ Rx_i @ Rz_w


def orbital_position_3d(a, e, i, Omega, omega, M):

    E = kepler_solver(M, e)

    # True anomaly
    nu = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )

    # Radius
    r = a * (1 - e * np.cos(E))

    # Position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    z_orb = 0

    r_vec_orb = np.array([x_orb, y_orb, z_orb])

    # Rotate to inertial frame
    R = rotation_matrix(Omega, i, omega)
    r_vec = R @ r_vec_orb

    return r_vec