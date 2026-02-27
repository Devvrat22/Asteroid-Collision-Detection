import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt



from physics import rk4

# constants reused by the notebook & dataset generator
DT = 3600  # 1 hour time step
AU = rk4.AU

# we rely on the helper functions defined in physics/rk4.py
earth_position = rk4.earth_position
acceleration = rk4.acceleration
rk4_step = rk4.rk4_step




def generate_transition_dataset(samples=50000):

    X = []
    Y = []

    for _ in range(samples):

        r = np.array([
            np.random.uniform(0.5, 2.0) * AU,
            np.random.uniform(-0.5, 0.5) * AU,
            np.random.uniform(-0.2, 0.2) * AU
        ])

        v = np.array([
            np.random.uniform(-35, 35),
            np.random.uniform(-35, 35),
            np.random.uniform(-10, 10)
        ])

        r_earth = earth_position(0)

        r_next, v_next = rk4_step(r, v, DT, r_earth)

        # NORMALIZE
        r_scaled = r / AU
        v_scaled = v / 40

        r_next_scaled = r_next / AU
        v_next_scaled = v_next / 40

        X.append(np.concatenate([r_scaled, v_scaled]))
        Y.append(np.concatenate([r_next_scaled, v_next_scaled]))

    return np.array(X), np.array(Y)



# the network class is defined in the physics package so that both
# training and the Streamlit/CLI code can import the same definition.
from physics.neural import NeuralIntegrator



def train_model():

    X, Y = generate_transition_dataset(40000)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    dataset = TensorDataset(X, Y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    model = NeuralIntegrator()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(20):

        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += loss_fn(model(xb), yb).item()

        print(f"Epoch {epoch+1} | Train Loss: {total_loss:.6f} | Val Loss: {val_loss:.6f}")

    torch.save(model.state_dict(), "neural_integrator.pt")

    return model



def rollout_test(model):

    steps = 1000

    r = np.array([1.2 * AU, 0, 0])
    v = np.array([0, 29, 5])

    r_true = r.copy()
    v_true = v.copy()

    r_pred = r.copy()
    v_pred = v.copy()

    true_traj = []
    pred_traj = []

    for step in range(steps):

        r_earth = earth_position(step * DT)

        # TRUE RK4
        r_true, v_true = rk4_step(r_true, v_true, DT, r_earth)

        # NEURAL
        state_scaled = np.concatenate([r_pred/AU, v_pred/40])
        state_tensor = torch.tensor(state_scaled, dtype=torch.float32)

        with torch.no_grad():
            next_state = model(state_tensor).numpy()

        r_pred = next_state[:3] * AU
        v_pred = next_state[3:] * 40

        true_traj.append(r_true.copy())
        pred_traj.append(r_pred.copy())

    true_traj = np.array(true_traj)
    pred_traj = np.array(pred_traj)

    error = np.linalg.norm(true_traj - pred_traj, axis=1)

    plt.plot(error)
    plt.title("Trajectory Divergence (km)")
    plt.xlabel("Step")
    plt.ylabel("Position Error (km)")
    plt.show()




if __name__ == "__main__":

    model = train_model()
    rollout_test(model)