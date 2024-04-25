import gymnasium as gym
import tinycarlo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List
from tinycarlo.helper import getenv

import os, sys, math
import numpy as np
from tqdm import trange

from examples.models.tinycar_net import TinycarCombo
from tinycarlo.wrapper import CTETerminationWrapper
from examples.benchmark_tinycar_net import pre_obs, evaluate

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
STEPS = 10000

### Data Collection
STEP_SIZE = 1000 # number of steps to take per episode
SKIP_STEPS = 2
BUFFER_SIZE = 25_000 # change this if you have not enough memory (150_000 = 27 GB)
BUFFER_SAVEFILE = "/tmp/stanley_training_data.npz" if len(sys.argv) != 2 else sys.argv[1]
MODEL_SAVEFILE = "/tmp/tinycar_combo.pt" if len(sys.argv) != 3 else sys.argv[2] 

ENV_SEED = 2
SPEED = 0.5
K = 5

NOISE_THETA = 0.1
NOISE_MEAN = 0.0
NOISE_SIGMA = 0.4 # we put in a lot of noise so network is more robust

action_dim = 1

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

noise = np.zeros(action_dim)

def create_loss_graph(loss: List[float]):
    import matplotlib.pyplot as plt
    plt.plot(loss)
    plt.xlabel("Steps * 10")
    plt.ylabel("Loss")
    plt.savefig("/tmp/stanley_loss.png")

def sample_episode(Xn, Mn, Yn, old_steps, env, maneuver = 0, seed = 0):
    global noise
    # random camera params
    pitch = np.random.randint(10, 20)
    fov = np.random.randint(90, 130)
    env.unwrapped.camera.orientation[0] = pitch
    env.unwrapped.camera.fov = fov
    env.unwrapped.camera.update_params()

    obs, info = env.reset(seed=seed)
    steps = 0
    for i in range(STEP_SIZE*SKIP_STEPS):
        cte, heading_error = info["cte"], info["heading_error"]
        # Lateral Control with Stanley Controller
        steering_correction = math.atan2(K * cte, SPEED)
        steering_angle = (heading_error + steering_correction) * 180 / math.pi / env.unwrapped.config["car"]["max_steering_angle"]
        noise += NOISE_THETA * (NOISE_MEAN - noise) + NOISE_SIGMA * np.random.randn(action_dim) # Ornstein-Uhlenbeck process
        action = {"car_control": [SPEED, steering_angle + noise[0]], "maneuver": maneuver if maneuver != 2 else 3}
        env.unwrapped.no_observation = False if (i+1)%SKIP_STEPS == 0 else True
        next_obs, _, terminated, truncated, info = env.step(action)
        if i%SKIP_STEPS == 0: # collect every SKIP_STEPS step
            Xn[old_steps+steps] = pre_obs(obs)
            Mn[old_steps+steps] = maneuver
            Yn[old_steps+steps] = steering_angle
            steps += 1
        obs = next_obs
        if terminated or truncated:
            break
    env.unwrapped.no_observation = False
    return steps
    
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config_simple_layout.yaml")
    env = gym.make("tinycarlo-v2", config=config_path)
    env = CTETerminationWrapper(env, max_cte=0.15)

    obs = pre_obs(env.reset(seed=ENV_SEED)[0]) # seed the environment and get obs shape

    tinycar_combo = TinycarCombo(obs.shape)
    opt = optim.Adam(tinycar_combo.parameters(), lr=LEARNING_RATE)
 
    print(f"using Device: {device} | encoder params: {sum([p.numel() for p in tinycar_combo.encoder.parameters()]):,} | actor params: {sum([p.numel() for p in tinycar_combo.actor.parameters()]):,}")
    # check if training data exists
    if os.path.exists(BUFFER_SAVEFILE):
        print(f"Loading training data from disk: {BUFFER_SAVEFILE}")
        data = np.load(BUFFER_SAVEFILE)
        Xn, Mn, Yn = data["Xn"], data["Mn"], data["Yn"]
        steps = Xn.shape[0]
    else:
        print("Collecting training data:")
        Xn, Mn, Yn = np.zeros((BUFFER_SIZE, *obs.shape), dtype=np.float32), np.zeros(BUFFER_SIZE, dtype=np.float32), np.zeros((BUFFER_SIZE, tinycar_combo.a_dim), dtype=np.float32)
        steps = 0 # buffer index
        maneuver, seed = 0, 0
        for episode_number in (t:=trange(BUFFER_SIZE//STEP_SIZE)):
            steps += sample_episode(Xn, Mn, Yn, steps, env, maneuver, seed)
            maneuver = (maneuver + 1) % 3
            if episode_number % 3 == 0: seed += 1
            t.set_description(f"sz: {steps:5d}")
        # save to disk
        Xn, Mn, Yn = Xn[:steps], Mn[:steps], Yn[:steps]
        np.savez_compressed(BUFFER_SAVEFILE, Xn=Xn, Mn=Mn, Yn=Yn)
    print(f"Training data memory used: {sum([x.size * x.itemsize for x in [Xn, Mn, Yn]])/1e9:.2f} GB | type: {Xn.dtype} | shape: {Xn.shape}")
    print("Training:")
    
    def train_step(x, m, y):
        opt.zero_grad()
        out = tinycar_combo(x, m)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        return loss
    
    tinycar_combo.to(device)
    losses, loss, loss_mean = [], 0, float("inf")
    for step in (t:=trange(STEPS)):
        samples = np.random.randint(0, steps, BATCH_SIZE)
        m = F.one_hot(torch.from_numpy(Mn[samples]).long(), num_classes=tinycar_combo.m_dim).float().to(device)
        loss += train_step(torch.from_numpy(Xn[samples]).to(device), m, torch.from_numpy(Yn[samples]).to(device)).item()
        if step % 10 == 0:
            loss_mean = loss / 10
            loss = 0
            losses.append(loss_mean)
        t.set_description(f"loss: {loss_mean:.7f}")

    create_loss_graph(losses[1:])
    
    print(f"Saving model to: {MODEL_SAVEFILE}")
    torch.save(tinycar_combo.state_dict(), MODEL_SAVEFILE)
    print("Evaluating:")
    for maneuver in range(3):
        eval_dict = evaluate(tinycar_combo, env.unwrapped, maneuver=maneuver, steps=1000, episodes=5, render_mode=None)
        cte_avg, cte_var, heading_error_avg, heading_error_var, terminations, steps_per_s, rew = eval_dict["cte_avg"], eval_dict["cte_var"], eval_dict["heading_error_avg"], eval_dict["heading_error_var"], eval_dict["terminations"], eval_dict["steps_per_s"], eval_dict["total_reward"] 
        print(f"Maneuver {maneuver} -> Total reward: {rew:.2f} | CTE: {cte_avg:.4f} m/step var: {cte_var:.4f}| Heading Error: {heading_error_avg:.4f} rad/step var {heading_error_var:.4f} | Terminations: {terminations:3d} | perf: {steps_per_s:.2f} steps/s")


