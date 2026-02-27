import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from physics.orbit import orbital_position_3d

# --- constants for two-body (Kepler) simulation ---
AU_TO_KM = 149_597_870.7

# --- constants for gravitational RK4 integration ---
G = 6.67430e-20  # km^3/kg/s^2
M_SUN = 1.989e30
M_EARTH = 5.972e24
AU = 149_597_870.7  # km


# ---------------------------------------------------------------------------
# Kepler two-body utilities (originally in kepler_twobodysim.py)
# ---------------------------------------------------------------------------

def run_kepler_simulation():
    """Simple Earth/asteroid two--body orbit plot based on orbital elements."""

    mu = 1.0

    # Earth elements
    earth = {"a": 1.0, "e": 0.0167, "i": 0.0, "Omega": 0.0, "omega": 0.0}

    # Asteroid elements
    asteroid = {
        "a": 1.2,
        "e": 0.4,
        "i": np.radians(20),
        "Omega": np.radians(40),
        "omega": np.radians(60),
    }

    n_earth = np.sqrt(mu / earth["a"] ** 3)
    n_ast = np.sqrt(mu / asteroid["a"] ** 3)

    times = np.linspace(0, 20 * np.pi, 1500)

    earth_positions = []
    ast_positions = []
    distances = []

    for t in times:
        M_earth = (n_earth * t) % (2 * np.pi)
        M_ast = (n_ast * t) % (2 * np.pi)

        r_earth = orbital_position_3d(
            earth["a"], earth["e"], earth["i"], earth["Omega"], earth["omega"], M_earth
        )

        r_ast = orbital_position_3d(
            asteroid["a"], asteroid["e"], asteroid["i"], asteroid["Omega"], asteroid["omega"], M_ast
        )

        dist = np.linalg.norm(r_ast - r_earth)

        earth_positions.append(r_earth)
        ast_positions.append(r_ast)
        distances.append(dist)

    earth_positions = np.array(earth_positions)
    ast_positions = np.array(ast_positions)
    distances = np.array(distances)

    # Time of closest approach
    idx_min = np.argmin(distances)
    min_distance_au = distances[idx_min]
    min_distance_km = min_distance_au * AU_TO_KM

    print("Closest approach time index:", idx_min)
    print(f"Minimum distance: {min_distance_au:.6f} AU")
    print(f"Minimum distance: {min_distance_km:.2f} km")

    animate_kepler_orbits(earth_positions, ast_positions)


def animate_kepler_orbits(earth_positions, ast_positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    earth_point, = ax.plot([], [], [], "bo")
    ast_point, = ax.plot([], [], [], "ro")

    ax.scatter(0, 0, 0)

    def update(frame):
        earth_point.set_data([earth_positions[frame, 0]], [earth_positions[frame, 1]])
        earth_point.set_3d_properties([earth_positions[frame, 2]])

        ast_point.set_data([ast_positions[frame, 0]], [ast_positions[frame, 1]])
        ast_point.set_3d_properties([ast_positions[frame, 2]])

        return earth_point, ast_point

    ani = FuncAnimation(fig, update, frames=len(earth_positions), interval=10)
    plt.show()


def generate_kepler_dataset(samples=300):
    """Randomly sample orbital elements and compute closest approach."""

    mu = 1.0
    dataset = []

    for _ in range(samples):
        # Random asteroid elements
        a = np.random.uniform(0.8, 2.5)
        e = np.random.uniform(0.0, 0.8)
        i = np.random.uniform(0, np.pi / 2)
        Omega = np.random.uniform(0, 2 * np.pi)
        omega = np.random.uniform(0, 2 * np.pi)

        n = np.sqrt(mu / a ** 3)
        times = np.linspace(0, 10 * np.pi, 400)

        min_dist = 999

        for t in times:
            M_ast = (n * t) % (2 * np.pi)
            M_earth = t % (2 * np.pi)

            r_ast = orbital_position_3d(a, e, i, Omega, omega, M_ast)
            r_earth = orbital_position_3d(1.0, 0.0167, 0, 0, 0, M_earth)

            dist = np.linalg.norm(r_ast - r_earth)

            if dist < min_dist:
                min_dist = dist

        collision = 1 if min_dist < 0.02 else 0

        dataset.append([a, e, i, Omega, omega, min_dist * AU_TO_KM, collision])

    return np.array(dataset)


# ---------------------------------------------------------------------------
# Gravitational RK4 and three–body utilities (from three_body.py / training code)
# ---------------------------------------------------------------------------

def earth_position(t):
    """Earth position using simple circular orbit about the sun."""
    omega = 2 * np.pi / (365.25 * 24 * 3600)
    x = AU * np.cos(omega * t)
    y = AU * np.sin(omega * t)
    z = 0.0
    return np.array([x, y, z])


def acceleration(r_ast, r_earth):
    """Acceleration on asteroid due to Sun (at origin) and Earth."""
    r_sun = np.zeros(3)
    r_as = r_ast - r_sun
    a_sun = -G * M_SUN * r_as / np.linalg.norm(r_as) ** 3

    r_ae = r_ast - r_earth
    a_earth = -G * M_EARTH * r_ae / np.linalg.norm(r_ae) ** 3

    return a_sun + a_earth


def rk4_step(r, v, dt, r_earth):
    """Perform one RK4 integration step for the asteroid."""

    def f_r(v):
        return v

    def f_v(r):
        return acceleration(r, r_earth)

    k1_r = f_r(v) * dt
    k1_v = f_v(r) * dt

    k2_r = f_r(v + 0.5 * k1_v) * dt
    k2_v = f_v(r + 0.5 * k1_r) * dt

    k3_r = f_r(v + 0.5 * k2_v) * dt
    k3_v = f_v(r + 0.5 * k2_r) * dt

    k4_r = f_r(v + k3_v) * dt
    k4_v = f_v(r + k3_r) * dt

    r_new = r + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
    v_new = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

    return r_new, v_new


def run_gravity_simulation():
    """Example driver for the restricted 3‑body RK4 simulation."""
    dt = 3600  # 1 hour
    total_time = 365 * 24 * 3600  # 1 year

    # Initial asteroid state (km, km/s)
    r_ast = np.array([1.2 * AU, 0, 0])
    v_ast = np.array([0, 29.0, 5.0])

    trajectory = []
    earth_traj = []
    distances = []

    t = 0
    while t < total_time:
        r_earth = earth_position(t)
        trajectory.append(r_ast.copy())
        earth_traj.append(r_earth.copy())

        dist = np.linalg.norm(r_ast - r_earth)
        distances.append(dist)

        r_ast, v_ast = rk4_step(r_ast, v_ast, dt, r_earth)
        t += dt

    trajectory = np.array(trajectory)
    earth_traj = np.array(earth_traj)
    distances = np.array(distances)

    idx_min = np.argmin(distances)
    min_distance_km = distances[idx_min]

    print("Closest approach index:", idx_min)
    print(f"Minimum distance: {min_distance_km:.2f} km")

    animate_gravity_orbits(earth_traj, trajectory)


def animate_gravity_orbits(earth_positions, ast_positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    scale = 2 * AU
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)

    earth_point, = ax.plot([], [], [], "bo")
    ast_point, = ax.plot([], [], [], "ro")

    ax.scatter(0, 0, 0, color="yellow", s=200)

    def update(frame):
        earth_point.set_data(
            [earth_positions[frame, 0]], [earth_positions[frame, 1]]
        )
        earth_point.set_3d_properties([earth_positions[frame, 2]])

        ast_point.set_data([ast_positions[frame, 0]], [ast_positions[frame, 1]])
        ast_point.set_3d_properties([ast_positions[frame, 2]])

        return earth_point, ast_point

    ani = FuncAnimation(fig, update, frames=len(earth_positions), interval=5)
    plt.show()
