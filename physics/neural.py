import torch
import torch.nn as nn
import numpy as np


class NeuralIntegrator(nn.Module):
    """Simple feed‑forward network that advances asteroid state one RK4 step.

    The architecture mirrors the one used in the training script; during
    inference the saved state dict is loaded and the network propogates
    normalized position/velocity vectors.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )

    def forward(self, x):
        return self.net(x)


def load_integrator(path: str = "models/neural_integrator.pt") -> NeuralIntegrator:
    """Instantiate a network and load weights from disk."""
    model = NeuralIntegrator()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def rollout(model: NeuralIntegrator, steps: int = 1000, dt: float = 3600.0):
    """Demonstration rollout comparing the learned integrator with the RK4
    ground truth.  Returns (true_traj, pred_traj, errors) arrays.
    """
    from physics.rk4 import rk4_step, earth_position, AU

    r = np.array([1.2 * AU, 0, 0])
    v = np.array([0, 29, 5])

    r_true = r.copy()
    v_true = v.copy()

    r_pred = r.copy()
    v_pred = v.copy()

    true_traj = []
    pred_traj = []

    for step in range(steps):
        r_earth = earth_position(step * dt)

        r_true, v_true = rk4_step(r_true, v_true, dt, r_earth)

        state_scaled = np.concatenate([r_pred / AU, v_pred / 40])
        state_tensor = torch.tensor(state_scaled, dtype=torch.float32)

        with torch.no_grad():
            next_state = model(state_tensor).numpy()

        r_pred = next_state[:3] * AU
        v_pred = next_state[3:] * 40

        true_traj.append(r_true.copy())
        pred_traj.append(r_pred.copy())

    tr = np.array(true_traj)
    pr = np.array(pred_traj)
    err = np.linalg.norm(tr - pr, axis=1)
    return tr, pr, err
