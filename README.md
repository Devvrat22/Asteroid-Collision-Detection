# Asteroid Collision Detection

Interactive asteroid trajectory simulation project combining:
- classical orbital mechanics (Kepler and RK4),
- Earth-asteroid distance analysis,
- a neural-network integrator benchmarked against RK4.

## What This Project Does

- Simulates asteroid motion with a restricted three-body RK4 model (Sun + Earth).
- Simulates Keplerian orbits from orbital elements.
- Compares a trained neural integrator against RK4 trajectories.
- Visualizes trajectories in 3D with Streamlit + Plotly.

## Project Layout

```
Asteroid_detection/
|-- app.py                      # Streamlit app (main interface)
|-- main.py                     # CLI entry point
|-- requirements.txt
|-- dataset.csv
|-- neural_integrator.pt        # Optional model weights location
|-- physics/
|   |-- rk4.py                  # Physics constants + RK4 simulation
|   |-- kepler_solver.py        # Newton solver for Kepler's equation
|   |-- orbit.py                # Keplerian 3D orbital position utilities
|   |-- neural.py               # Neural integrator model + rollout
|   |-- collision.py
|-- models/
|   |-- train_neural_generator.py
|   |-- *.ipynb
|-- notebooks/
|   |-- *.ipynb
|-- tests/
|   |-- test_imports.py
```

## Streamlit Modes

The app currently exposes three modes:

1. `Gravitational RK4`
- Integrates asteroid state with RK4 under Sun + Earth gravity.
- Reports minimum Earth-asteroid distance.
- Plots 3D Earth and asteroid trajectories.

2. `Kepler simulation`
- Uses orbital elements (`a, e, i, Ω, ω`) and mean anomaly progression.
- Computes asteroid and Earth Keplerian trajectories.
- Reports minimum Earth-asteroid distance in this Kepler model.

3. `Neural rollout`
- Loads `NeuralIntegrator` PyTorch weights.
- Rolls out predicted trajectory and compares against RK4.
- Plots prediction error and 3D trajectory comparison.

## Installation

### 1. Create and activate a virtual environment

PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Git Bash:
```bash
python -m venv venv
source venv/Scripts/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the App

From project root:

PowerShell:
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Git Bash:
```bash
./venv/Scripts/python.exe -m streamlit run app.py
```

This `python -m streamlit` form is recommended when `streamlit` is not on PATH.

## Neural Model Weights

`physics/neural.py` looks for weights in this order:

1. `models/neural_integrator.pt`
2. `<project_root>/models/neural_integrator.pt`
3. `<project_root>/neural_integrator.pt`

If not found, the app shows a clear error in Neural mode instead of crashing.

## Troubleshooting

1. `missing ScriptRunContext`
- Cause: running app with `python app.py`.
- Fix: run with `python -m streamlit run app.py`.

2. `streamlit: command not found` / `not recognized`
- Cause: Streamlit script path not in PATH.
- Fix: use `python -m streamlit run app.py` (or `py -m streamlit run app.py`).

3. `FileNotFoundError: models/neural_integrator.pt`
- Cause: weights file not at expected path.
- Fix: place file at one of the paths listed in `Neural Model Weights`.

## Development Notes

- Physics units are primarily km, kg, s in `physics/rk4.py`.
- Earth orbit in RK4 mode uses a circular approximation.
- Some notebooks and model experiments are included for offline analysis/training.

## Testing

```bash
pytest tests
```

