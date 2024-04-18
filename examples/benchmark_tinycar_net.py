import gymnasium as gym
from typing import Tuple
from tinycarlo.wrapper import CTELinearRewardWrapper, LanelineSparseRewardWrapper, CTETerminationWrapper
import tinycarlo
from tinycarlo.helper import getenv

import os, sys, time
import numpy as np

from examples.models.tinycar_net import TinycarCombo, TinycarEncoder, TinycarActorTemporal, TinycarActor
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

ENV_SEED = 10
ACTOR = getenv("ACTOR") # if set, pretrained tinycombo is loaded and the provided actor swapped in
TEMPORAL = getenv("TEMPORAL") # if set, actor is TinycarActorTemporal

def pre_obs(obs: np.ndarray) -> np.ndarray:
    # cropping and normalizing the image
    #return np.stack([obs[i,obs.shape[1]//2:,:]/255 for i in range(obs.shape[0])], axis=0).astype(np.float32)
    return (obs/255).astype(np.float32)

def evaluate(model: TinycarCombo, unwrapped_env: gym.Env, maneuver: int, seed: int = 0, speed = 0.3, steps = 5000, episodes = 5, render_mode=None, temporal: int = 1) -> Tuple[float, float, float, int, float]:
    """
    Tests the model in the environment for a given maneuver.
    Returns total reward, average CTE, and average heading error
    """
    unwrapped_env.unwrapped.render_mode = render_mode
    model.to(device)
    model.eval()

    env = CTELinearRewardWrapper(unwrapped_env, min_cte=0.03, max_reward=1.0)
    env = LanelineSparseRewardWrapper(env, sparse_rewards={"outer": -10.0})
    env = CTETerminationWrapper(env, max_cte=0.1)

    def get_steering_angle(x, m, seq_x):
        with torch.no_grad():
            if temporal > 1:
                seq_x = seq_x.roll(1, 0)
                seq_x[0] = model.encoder(x.unsqueeze(0))[0]
                out = model.actor(seq_x.unsqueeze(0), m)[0].cpu().item()
            else:
                out = model.forward(x.unsqueeze(0), m)[0].cpu().item()
        return out

    obs = env.reset(seed=seed)[0]
    total_rew, cte, heading_error, terminations, inf_time = 0.0, [], [], 0, []
    terminated, truncated = False, False
    seq_x = torch.zeros(temporal, TinycarEncoder.FEATURE_VEC_SIZE).to(device)
    for i in range(int(steps * episodes)):
        st = time.perf_counter()
        x = torch.from_numpy(pre_obs(obs.astype(np.float32))).to(device)
        m = F.one_hot(torch.tensor(maneuver), num_classes=model.m_dim).float().unsqueeze(0).to(device)
        steering_angle = get_steering_angle(x, m, seq_x)
        inf_time.append(time.perf_counter() - st)
        obs, rew, terminated, truncated, info = env.step({"car_control": [speed, steering_angle], "maneuver": maneuver if maneuver != 2 else 3})
        total_rew += rew
        cte.append(abs(info["cte"]))
        heading_error.append(abs(info["heading_error"]))
        if terminated or truncated:
            terminations += 1
            obs = env.reset()[0]
        if i % steps == 0:
            obs = env.reset()[0]
    return total_rew, sum(cte) / len(cte), sum(heading_error) / len(heading_error), terminations, steps * episodes / sum(inf_time)
    
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config_simple_layout.yaml")
    env = gym.make("tinycarlo-v2", config=config_path)

    obs = pre_obs(env.reset(seed=ENV_SEED)[0]) # seed the environment and get obs shape

    tinycar_combo = TinycarCombo(obs.shape)
    tinycar_combo.load_pretrained(device)
    if len(sys.argv) == 2:
        if ACTOR:
            actor = TinycarActorTemporal(seq_len=10) if TEMPORAL else TinycarActor()
            actor.load_state_dict(torch.load(sys.argv[1]), strict=False)
            tinycar_combo.actor = actor
        else:
            tinycar_combo.load_state_dict(torch.load(sys.argv[1]), strict=False)

    for maneuver in range(3):
        rew, cte, heading_error, terminations, stepss = evaluate(tinycar_combo, env, maneuver=maneuver, steps=2000, episodes=5, render_mode="human", temporal=10 if TEMPORAL else 1, seed=ENV_SEED)
        print(f"Maneuver {maneuver} -> Total reward: {rew:.2f} | CTE: {cte:.4f} m/step | Heading Error: {heading_error:.4f} rad/step | Terminations: {terminations:3d} | perf: {stepss:.2f} steps/s")
    



    
