import streamlit as st
import numpy as np
import plotly.graph_objects as go

from physics.rk4 import simulate_orbit, AU
from physics.neural import load_integrator, rollout
from physics.orbit import orbital_position_3d


st.set_page_config(layout="wide")
st.title("Asteroid Detection Playground")

mode = st.sidebar.selectbox(
    "Select mode",
    ["Gravitational RK4", "Kepler simulation", "Neural rollout"]
)

# =====================================================
# RK4 SIMULATION MODE
# =====================================================

if mode == "Gravitational RK4":

    st.markdown("### Restricted Three-Body RK4 Integration")

    total_days = st.slider("Simulation days", 50, 800, 365)
    dt_hours = st.slider("Timestep (hours)", 1, 24, 6)

    if st.button("Run RK4 Simulation"):

        ast_pos, earth_pos, min_dist = simulate_orbit(
            total_days=total_days,
            dt_hours=dt_hours
        )

        st.success(f"Minimum distance: {min_dist:,.2f} km")

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=earth_pos[:, 0],
            y=earth_pos[:, 1],
            z=earth_pos[:, 2],
            mode="lines",
            name="Earth"
        ))

        fig.add_trace(go.Scatter3d(
            x=ast_pos[:, 0],
            y=ast_pos[:, 1],
            z=ast_pos[:, 2],
            mode="lines",
            name="Asteroid"
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                zaxis_title="Z (km)"
            )
        )

        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# KEPLER SIMULATION MODE
# =====================================================

elif mode == "Kepler simulation":

    st.markdown("### Keplerian Orbit Simulation")

    a_au = st.slider("Semi-major axis (AU)", 0.5, 5.0, 1.2, 0.05)
    e = st.slider("Eccentricity", 0.0, 0.95, 0.2, 0.01)
    i_deg = st.slider("Inclination i (deg)", 0.0, 180.0, 10.0, 0.5)
    Omega_deg = st.slider("Longitude of ascending node Ω (deg)", 0.0, 360.0, 80.0, 1.0)
    omega_deg = st.slider("Argument of periapsis ω (deg)", 0.0, 360.0, 40.0, 1.0)
    period_days = st.slider("Orbital period (days)", 30, 2000, 500)
    sim_days = st.slider("Simulation span (days)", 30, 2000, 730)
    points = st.slider("Points", 200, 4000, 1200, 100)

    if st.button("Run Kepler Simulation"):
        a_km = a_au * AU
        i = np.radians(i_deg)
        Omega = np.radians(Omega_deg)
        omega = np.radians(omega_deg)

        times = np.linspace(0.0, sim_days, points)
        mean_anomaly = 2 * np.pi * (times / period_days)

        asteroid_traj = np.array(
            [orbital_position_3d(a_km, e, i, Omega, omega, M) for M in mean_anomaly]
        )

        earth_M = 2 * np.pi * (times / 365.25)
        earth_traj = np.array(
            [orbital_position_3d(AU, 0.0167, 0.0, 0.0, 0.0, M) for M in earth_M]
        )

        distances = np.linalg.norm(asteroid_traj - earth_traj, axis=1)
        min_dist = float(np.min(distances))
        st.success(f"Minimum Earth-asteroid distance (Kepler model): {min_dist:,.2f} km")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=earth_traj[:, 0],
                y=earth_traj[:, 1],
                z=earth_traj[:, 2],
                mode="lines",
                name="Earth (Kepler)",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=asteroid_traj[:, 0],
                y=asteroid_traj[:, 1],
                z=asteroid_traj[:, 2],
                mode="lines",
                name="Asteroid (Kepler)",
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                zaxis_title="Z (km)",
            )
        )
        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# NEURAL ROLLOUT MODE
# =====================================================

elif mode == "Neural rollout":

    st.markdown("### Neural Integrator vs RK4")

    steps = st.slider("Steps", 100, 2000, 1000, step=100)

    if st.button("Run Neural Rollout"):
        try:
            model = load_integrator()
            true_traj, pred_traj, errors = rollout(model, steps=steps)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()

        st.line_chart(errors)
        st.write("Final error (km):", float(errors[-1]))

        # Optional 3D comparison
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=true_traj[:, 0],
            y=true_traj[:, 1],
            z=true_traj[:, 2],
            mode="lines",
            name="True (RK4)"
        ))

        fig.add_trace(go.Scatter3d(
            x=pred_traj[:, 0],
            y=pred_traj[:, 1],
            z=pred_traj[:, 2],
            mode="lines",
            name="Neural Prediction"
        ))

        st.plotly_chart(fig, use_container_width=True)
