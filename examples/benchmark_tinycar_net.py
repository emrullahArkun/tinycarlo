import gymnasium as gym
from typing import Tuple
from tinycarlo.wrapper import CTELinearRewardWrapper, LanelineSparseRewardWrapper, CTETerminationWrapper, CrashTerminationWrapper
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
    return (obs/255).astype(np.float32)

def evaluate(model: TinycarCombo, unwrapped_env: gym.Env, maneuver: int, seed: int = 0, speed = 0.35, steps = 5000, episodes = 5, render_mode=None, temporal: int = 1) -> Tuple[float, float, float, int, float]:
    """
    Tests the model in the environment for a given maneuver.
    Returns total reward, average CTE, and average heading error
    """
    unwrapped_env.unwrapped.render_mode = render_mode
    model.to(device)
    model.eval()

    env = CTELinearRewardWrapper(unwrapped_env, min_cte=0.03, max_reward=1.0, min_reward=-1.0)
    env = CTETerminationWrapper(env, max_cte=0.1, number_of_steps=5)
    env = CrashTerminationWrapper(env)

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
    total_rew, cte, heading_error, terminations, inf_time, positions = 0.0, [], [], 0, [], []
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
        positions.append(info["position"])
        if terminated or truncated:
            if terminated:
                terminations += 1
            obs = env.reset()[0]
        if i % steps == 0:
            obs = env.reset()[0]
    # save positions for visualization in examples/render_map.py
    np.save(f"/tmp/positions_m{maneuver}.npy", np.array(positions))
    np.save(f"/tmp/cte_m{maneuver}.npy", np.array(cte))
    np.save(f"/tmp/heading_error_m{maneuver}.npy", np.array(heading_error))

    cte_avg = sum(cte) / len(cte)
    cte_var = sum((x - cte_avg) ** 2 for x in cte) / len(cte)
    heading_error_avg = sum(heading_error) / len(heading_error)
    heading_error_var = sum((x - heading_error_avg) ** 2 for x in heading_error) / len(heading_error)
    ret = {"cte_avg": cte_avg, "cte_var": cte_var, "heading_error_avg": heading_error_avg, "heading_error_var": heading_error_var, "terminations": terminations, "steps_per_s": steps * episodes / sum(inf_time), "total_reward": total_rew}
    return ret
    
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_simple_layout.yaml")
    env = gym.make("tinycarlo-realworld-v2", config=config_path, render_mode="human")

    obs = pre_obs(env.reset(seed=ENV_SEED)[0]) # seed the environment and get obs shape

    tinycar_combo = TinycarCombo(obs.shape)
    tinycar_combo.load_pretrained(device)
    if len(sys.argv) == 2:
        if ACTOR:
            actor = TinycarActorTemporal(seq_len=10) if TEMPORAL else TinycarActor()
            actor.load_state_dict(torch.load(sys.argv[1], map_location=device), strict=False)
            tinycar_combo.actor = actor
        else:
            tinycar_combo.load_state_dict(torch.load(sys.argv[1], map_location=device), strict=False)

    for maneuver in range(3):
        eval_dict = evaluate(tinycar_combo, env, maneuver=maneuver, steps=1000, episodes=5, render_mode=None, temporal=10 if TEMPORAL else 1, seed=ENV_SEED)
        cte_avg, cte_var, heading_error_avg, heading_error_var, terminations, steps_per_s, rew = eval_dict["cte_avg"], eval_dict["cte_var"], eval_dict["heading_error_avg"], eval_dict["heading_error_var"], eval_dict["terminations"], eval_dict["steps_per_s"], eval_dict["total_reward"] 
        print(f"Maneuver {maneuver} -> Total reward: {rew:.2f} | CTE: {cte_avg:.4f} m/step var: {cte_var:.4f}| Heading Error: {heading_error_avg:.4f} rad/step var {heading_error_var:.4f} | Terminations: {terminations:3d} | perf: {steps_per_s:.2f} steps/s")
    



    
