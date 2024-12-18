import csv
from typing import List, Tuple
import torch
import torch.nn.functional as F
import numpy as np

#####################
##### Only utils for reinforcement learning examples
if __name__ == "__main__": print("This script is not meant to be run directly. Run the examples instead.")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class Replaybuffer:
    def __init__(self, size: int, batch_size: int, obs_shape: List[int], maneuver_dim: int, action_dim: int) -> None:
        self.size, self.batch_size, self.obs_shape, self.maneuver_dim, self.action_dim = size, batch_size, obs_shape, maneuver_dim, action_dim
        self.X, self.M, self.A, self.R, self.X1 = np.zeros((size, *obs_shape), dtype=np.float32), np.zeros( (size,), dtype=np.int64), np.zeros((size, self.action_dim), dtype=np.float32), np.zeros((size, 1), dtype=np.float32), np.zeros( (size, *obs_shape), dtype=np.float32)
        self.rp_sz = 0

    def add(self, x, m, a, r, x1) -> None:
        rp_idx = self.rp_sz if self.rp_sz < self.size else np.random.randint(0, self.size)
        self.X[rp_idx], self.M[rp_idx], self.A[rp_idx], self.R[rp_idx], self.X1[rp_idx] = x, m, a, r, x1
        self.rp_sz = min(self.rp_sz + 1, self.size)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.rp_sz >= self.batch_size
        x, m, a, r, x1, m1 = self[np.random.randint(0, self.rp_sz, self.batch_size)]
        return torch.from_numpy(x).to(device), F.one_hot(torch.from_numpy(m), self.maneuver_dim).float().to(device), torch.from_numpy(a).to(device), torch.from_numpy(r).to(device), torch.from_numpy(x1).to(device), F.one_hot(torch.from_numpy(m1), self.maneuver_dim).float().to(device)

    def __getitem__(self, indices):
        return self.X[indices], self.M[indices], self.A[indices], self.R[indices], self.X1[indices], self.M[indices]
    
class ReplaybufferTemporal:
    def __init__(self, size: int, batch_size: int, obs_shape: List[int], maneuver_dim: int, action_dim: int, seq_len: int) -> None:
        self.size, self.batch_size, self.obs_shape, self.maneuver_dim, self.action_dim, self.seq_len = size, batch_size, obs_shape, maneuver_dim, action_dim, seq_len
        self.X, self.M, self.A, self.R, self.X1 = np.zeros((size, seq_len, *obs_shape), dtype=np.float32), np.zeros( (size,), dtype=np.int64), np.zeros((size, self.action_dim), dtype=np.float32), np.zeros((size, 1), dtype=np.float32), np.zeros( (size, seq_len, *obs_shape), dtype=np.float32)
        self.rp_sz = 0

    def add(self, x, m, a, r, x1) -> None:
        rp_idx = self.rp_sz if self.rp_sz < self.size else np.random.randint(0, self.size)
        self.X[rp_idx], self.M[rp_idx], self.A[rp_idx], self.R[rp_idx], self.X1[rp_idx] = x, m, a, r, x1
        self.rp_sz = min(self.rp_sz + 1, self.size)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.rp_sz >= self.batch_size
        x, m, a, r, x1, m1 = self[np.random.randint(0, self.rp_sz, self.batch_size)]
        return torch.from_numpy(x).to(device), F.one_hot(torch.from_numpy(m), self.maneuver_dim).float().to(device), torch.from_numpy(a).to(device), torch.from_numpy(r).to(device), torch.from_numpy(x1).to(device), F.one_hot(torch.from_numpy(m1), self.maneuver_dim).float().to(device)
    
    def save_to_disk(self, filename: str) -> None:
        np.savez_compressed(filename, X=self.X, M=self.M, A=self.A, R=self.R, X1=self.X1)

    def load_from_disk(self, filename: str) -> None:
        data = np.load(filename)
        self.X, self.M, self.A, self.R, self.X1 = data["X"], data["M"], data["A"], data["R"], data["X1"]
        self.rp_sz = self.X.shape[0]

    def __getitem__(self, indices):
        return self.X[indices], self.M[indices], self.A[indices], self.R[indices], self.X1[indices], self.M[indices]

def create_critic_loss_graph(c1_loss: List[float], c2_loss: List[float], suffix: str):
    import matplotlib.pyplot as plt
    # Plot erstellen ohne gleitenden Durchschnitt
    plt.figure()
    plt.plot(c1_loss)
    plt.plot(c2_loss)
    plt.legend(["Critic 1 Loss", "Critic 2 Loss"])
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Critic Loss over Steps")
    plt.savefig("/home/emrullah/Schreibtisch/critic_loss.png")  # Bild speichern

    # CSV speichern
    with open(f"domain_randomization/data/critic_loss_{suffix}", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Critic 1 Loss", "Critic 2 Loss"])  # Header
        for step, (loss1, loss2) in enumerate(zip(c1_loss, c2_loss), start=1):
            writer.writerow([step, loss1, loss2])  # Zeilen mit Schritt und beiden Loss-Werten schreiben

def create_action_loss_graph(a_loss: List[float], suffix: str):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(ma(a_loss))
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("/home/emrullah/Schreibtisch/actor_loss.png")

    # CSV speichern
    with open(f"domain_randomization/data/actor_loss_{suffix}", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "Loss"])  # Header
        for step, loss in enumerate(a_loss, start=1):
            writer.writerow([step, loss])

def create_ep_rew_graph(ep_rews: List[float], suffix: str):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(ma(ep_rews, 10))
    plt.xlabel("Episodes")
    plt.ylabel("Episodic Reward")
    plt.savefig("/home/emrullah/Schreibtisch/ep_rew.png")

    # CSV speichern
    with open(f"domain_randomization/data/rew_{suffix}", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Reward"])  # Header
        for episode, reward in enumerate(ep_rews, start=1):
            writer.writerow([episode, reward])

def avg_w(x: List[float], w: int = 100) -> List[float]:
    if len(x) < w:
        return float("inf")
    return sum(x[-w:]) / w

def ma(x: List[float], w: int = 100) -> List[float]:
    return [sum(x[i:i+w])/w for i in range(len(x)-w)]