import numpy as np

# =============================
# Physical Constants (km, kg, s)
# =============================

G = 6.67430e-20          # km^3 / kg / s^2
M_SUN = 1.989e30         # kg
M_EARTH = 5.972e24       # kg
AU = 149_597_870.7       # km


# =============================
# Earth Circular Orbit Model
# =============================

def earth_position(t):
    """
    Returns Earth's position around Sun (circular orbit approximation).
    t in seconds.
    """
    omega = 2 * np.pi / (365.25 * 24 * 3600)  # rad/s
    x = AU * np.cos(omega * t)
    y = AU * np.sin(omega * t)
    z = 0.0
    return np.array([x, y, z])


# =============================
# Gravitational Acceleration
# =============================

def acceleration(r_ast, r_earth):
    """
    Acceleration of asteroid due to Sun (origin) and Earth.
    """
    # Sun contribution
    r_sun = np.zeros(3)
    r_as = r_ast - r_sun
    a_sun = -G * M_SUN * r_as / np.linalg.norm(r_as)**3

    # Earth contribution
    r_ae = r_ast - r_earth
    a_earth = -G * M_EARTH * r_ae / np.linalg.norm(r_ae)**3

    return a_sun + a_earth


# =============================
# RK4 Integrator Step
# =============================

def rk4_step(r, v, dt, t):
    """
    One RK4 integration step.
    """

    r_earth = earth_position(t)

    def f_r(v_local):
        return v_local

    def f_v(r_local):
        return acceleration(r_local, r_earth)

    k1_r = f_r(v) * dt
    k1_v = f_v(r) * dt

    k2_r = f_r(v + 0.5 * k1_v) * dt
    k2_v = f_v(r + 0.5 * k1_r) * dt

    k3_r = f_r(v + 0.5 * k2_v) * dt
    k3_v = f_v(r + 0.5 * k2_r) * dt

    k4_r = f_r(v + k3_v) * dt
    k4_v = f_v(r + k3_r) * dt

    r_new = r + (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6
    v_new = v + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6

    return r_new, v_new


# =============================
# Main Simulation Function
# =============================

def simulate_orbit(
    total_days=365,
    dt_hours=6,
    initial_position=None,
    initial_velocity=None
):
    """
    Runs full RK4 simulation and returns:
    - asteroid trajectory
    - earth trajectory
    - minimum distance (km)
    """

    dt = dt_hours * 3600
    total_time = total_days * 24 * 3600

    # Default asteroid initial state
    if initial_position is None:
        r_ast = np.array([1.2 * AU, 0, 0])
    else:
        r_ast = np.array(initial_position)

    if initial_velocity is None:
        v_ast = np.array([0, 29.0, 5.0])
    else:
        v_ast = np.array(initial_velocity)

    trajectory = []
    earth_traj = []
    distances = []

    t = 0.0

    while t < total_time:
        r_earth = earth_position(t)

        trajectory.append(r_ast.copy())
        earth_traj.append(r_earth.copy())

        dist = np.linalg.norm(r_ast - r_earth)
        distances.append(dist)

        r_ast, v_ast = rk4_step(r_ast, v_ast, dt, t)
        t += dt

    trajectory = np.array(trajectory)
    earth_traj = np.array(earth_traj)
    distances = np.array(distances)

    min_distance = np.min(distances)

    return trajectory, earth_traj, min_distance