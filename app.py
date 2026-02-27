import streamlit as st
import numpy as np

from physics import rk4
from physics.neural import load_integrator, rollout

st.title("Asteroid Detection Playground")

mode = st.sidebar.selectbox("Select mode", ["Kepler orbit", "Gravitational RK4", "Neural rollout"])

if mode == "Kepler orbit":
    st.markdown("### Two‑body Kepler simulation")
    if st.button("Run simulation"):
        rk4.run_kepler_simulation()

elif mode == "Gravitational RK4":
    st.markdown("### Restricted three‑body RK4 integration")
    if st.button("Run simulation"):
        rk4.run_gravity_simulation()

elif mode == "Neural rollout":
    st.markdown("### Compare trained neural integrator vs RK4")
    model = load_integrator()
    steps = st.slider("Steps", min_value=100, max_value=2000, value=1000, step=100)
    if st.button("Run rollout"):
        true_traj, pred_traj, errors = rollout(model, steps=steps)
        st.line_chart(errors)
        st.write("Final error (km):", float(errors[-1]))
