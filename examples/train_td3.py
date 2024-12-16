import gymnasium as gym
import imageio.v2 as imageio
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

import tinycarlo
from typing import Tuple, List
import torch
import torch.nn.functional as F

import os, sys
import numpy as np
from tqdm import trange
import time

from examples.domain_randomization.vis_utils import create_distance_graph, create_cte_graph, calculate_weight_changes, \
    plot_all_weight_changes
from tinycarlo.wrapper.reward import CTELinearRewardWrapper, LanelineSparseRewardWrapper
from tinycarlo.wrapper.termination import LanelineCrossingTerminationWrapper, CTETerminationWrapper, CrashTerminationWrapper
from examples.models.tinycar_net import TinycarActorTemporal, TinycarCriticTemporal, TinycarCombo, TinycarEncoder
from examples.benchmark_tinycar_net import pre_obs, evaluate
from tinycarlo.helper import getenv
from examples.rl_utils import avg_w, create_action_loss_graph, create_critic_loss_graph, create_ep_rew_graph, \
    Replaybuffer, ReplaybufferTemporal

# *** hyperparameters ***
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 500_000
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 2e-4
EPISODES = 100
DISCOUNT_FACTOR = 0.99
TAU = 0.001  # soft update parameter
POLICY_DELAY = 2  # Delayed policy updates
MAX_STEPS = 1000

# Shift Parameter
INCLUDE_SHIFT = True  # Global parameter for the shift in steering angle

# *** environment parameters ***
SPEED = 0.4

NOISE_THETA = 0.1
NOISE_MEAN = 0.0
NOISE_SIGMA = 0.4

SEQ_LEN = 10

STEERING_SHIFT = -0.001

MODEL_SAVEFILE = "/tmp/actor_td3.pt"
PRETRAINED_MODEL = sys.argv[1] if len(sys.argv) > 1 else None

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

if __name__ == "__main__":
    print(f"Shift in steering angle: {INCLUDE_SHIFT}")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config_simple_layout.yaml")
    env = gym.make("tinycarlo-v2", config=config_path, render_mode="human")

    env = CTELinearRewardWrapper(env, min_cte=0.03, max_reward=1.0, min_reward=-1.0)
    env = CTETerminationWrapper(env, max_cte=0.1, number_of_steps=5)
    env = CrashTerminationWrapper(env)

    obs = pre_obs(env.reset()[0])  # seed the environment and get obs shape
    tinycar_combo = TinycarCombo(obs.shape)
    tinycar_combo.load_pretrained(device)
    encoder = tinycar_combo.encoder
    actor = TinycarActorTemporal(seq_len=SEQ_LEN)
    actor_target = TinycarActorTemporal(seq_len=SEQ_LEN)
    if PRETRAINED_MODEL:
        actor.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=device))
    critic1 = TinycarCriticTemporal(seq_len=SEQ_LEN)
    critic2 = TinycarCriticTemporal(seq_len=SEQ_LEN)
    critic_target1 = TinycarCriticTemporal(seq_len=SEQ_LEN)
    critic_target2 = TinycarCriticTemporal(seq_len=SEQ_LEN)

    encoder.to(device)
    actor.to(device)
    actor_target.to(device)
    critic1.to(device)
    critic2.to(device)
    critic_target1.to(device)
    critic_target2.to(device)

    action_dim, maneuver_dim = tinycar_combo.a_dim, tinycar_combo.m_dim

    # load the same weights into the target networks initially
    for v, v_target in zip(actor.parameters(), actor_target.parameters()):
        v_target.data.copy_(v.data)
    for v, v_target in zip(critic1.parameters(), critic_target1.parameters()):
        v_target.data.copy_(v.data)
    for v, v_target in zip(critic2.parameters(), critic_target2.parameters()):
        v_target.data.copy_(v.data)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE_ACTOR)
    opt_critic1 = torch.optim.Adam(critic1.parameters(), lr=LEARNING_RATE_CRITIC)
    opt_critic2 = torch.optim.Adam(critic2.parameters(), lr=LEARNING_RATE_CRITIC)

    replay_buffer = ReplaybufferTemporal(REPLAY_BUFFER_SIZE, BATCH_SIZE, (TinycarEncoder.FEATURE_VEC_SIZE,), maneuver_dim, action_dim, SEQ_LEN)

    noise = torch.zeros(action_dim).to(device)

    print( f"using Device: {device} | actor params {sum([p.numel() for p in actor.parameters()]):,} | critic params {sum([p.numel() for p in critic1.parameters()]):,}")

    def train_step_critic(x, m, a, r, x1, m1):
        global noise
        with torch.no_grad():
           # target_action = (actor_target(x1, m1) + noise).clamp(-1, 1)
            target_action = actor_target(x1, m1)
            target_q1 = critic_target1(x1, m1, target_action)
            target_q2 = critic_target2(x1, m1, target_action)
            target_q = torch.min(target_q1,target_q2) * DISCOUNT_FACTOR + r

        # Update critic networks
        opt_critic1.zero_grad()
        opt_critic2.zero_grad()
        critic1_loss = F.mse_loss(target_q, critic1(x, m, a))
        critic2_loss = F.mse_loss(target_q, critic2(x, m, a))
        critic1_loss.backward()
        critic2_loss.backward()
        opt_critic1.step()
        opt_critic2.step()

        return critic1_loss, critic2_loss
    
    def train_step_action(x: torch.Tensor,m: torch.Tensor) -> torch.Tensor:
        opt_actor.zero_grad()
        actor_loss = -critic1(x, m, actor(x, m)).mean()
        actor_loss.backward()
        opt_actor.step()

        for v, v_target in zip(actor.parameters(), actor_target.parameters()):
            v_target.data.copy_(TAU * v.data + (1 - TAU) * v_target.data)
        for v, v_target in zip(critic1.parameters(), critic_target1.parameters()):
            v_target.data.copy_(TAU * v.data + (1 - TAU) * v_target.data)
        for v, v_target in zip(critic2.parameters(), critic_target2.parameters()):
            v_target.data.copy_(TAU * v.data + (1 - TAU) * v_target.data)
        return actor_loss

    def get_action(feature_vec: torch.Tensor, maneuver: torch.Tensor) -> torch.Tensor:
        global noise, exploration_rate, INCLUDE_SHIFT
        with torch.no_grad():
            noise += NOISE_THETA * (NOISE_MEAN - noise) + NOISE_SIGMA * torch.randn(action_dim).to(device)  # Ornstein-Uhlenbeck process
            action = actor(feature_vec.unsqueeze(0), maneuver.unsqueeze(0))[0] + noise
            #print(f"Action: {action} before adding noise")
            if INCLUDE_SHIFT:
                torch.add(action, STEERING_SHIFT, out=action)
            action = action.clamp(-1, 1).cpu()
            #print(f"Action: {action} after adding noise")
        return action
    
    def get_feature_vec(obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feature_vec = encoder(obs.unsqueeze(0))[0]
        return feature_vec


    def save_weights(network, step, weight_history):
        weight_history[step] = {name: param.clone().cpu().detach().numpy() for name, param in network.named_parameters() if 'weight' in name}


    st, steps = time.perf_counter(), 0
    rews = []
    c1_loss, c2_loss, a_loss = [],[],[]
    outer_dist, dashed_dist, solid_dist  = [],[],[]
    feature_vec_queue = torch.zeros(SEQ_LEN+1, TinycarEncoder.FEATURE_VEC_SIZE).to(device)
    ctes = []
    latent_space = []
    # Initialize a dictionary to store weights
    actor_weight_history = {}
    critic1_weight_history = {}
    critic2_weight_history = {}
    # Save initial weights
    save_weights(actor, 0, actor_weight_history)
    save_weights(actor, 0, critic1_weight_history)
    save_weights(actor, 0, critic2_weight_history)
    for episode_number in (t := trange(EPISODES)):
        feature_vec_queue = torch.roll(feature_vec_queue, 1, 0)
        feature_vec_queue[0] = get_feature_vec(torch.from_numpy(pre_obs(env.reset()[0])).to(device))
        noise = torch.zeros(action_dim).to(device) # reset the noise
        maneuver = np.random.randint(0, 3)
        NOISE_SIGMA = 0.4 * (1 - (episode_number / EPISODES))
        ep_rew = 0
        for ep_step in range(MAX_STEPS):
            if steps % 200 == 0:  # Save weights every 200 steps
                save_weights(actor, steps, actor_weight_history)
                save_weights(critic1, steps,critic1_weight_history)
                save_weights(critic2, steps, critic2_weight_history)
            m = F.one_hot(torch.tensor(maneuver), tinycar_combo.m_dim).float().to(device)
            act = get_action(feature_vec_queue[:-1,:], m).item()
            obs, rew, terminated, truncated, info = env.step({"car_control": [SPEED, act], "maneuver": maneuver if maneuver != 2 else 3})
            ctes.append(info["cte"])
            #print(actor.state_dict().keys())
            # Append the current feature vector to the list
            outer_dist.append(info["laneline_distances"]["solid"])
            dashed_dist.append(info["laneline_distances"]["dashed"])
            solid_dist.append(info["laneline_distances"]["solid"])
            feature_vec_queue = torch.roll(feature_vec_queue, 1, 0)
            feature_vec_queue[0] = get_feature_vec(torch.from_numpy(pre_obs(obs)).to(device))

            latent_space.append(get_feature_vec(torch.from_numpy(pre_obs(obs))).cpu().numpy())
            #print(get_feature_vec(torch.from_numpy(pre_obs(obs))).cpu().numpy())

            replay_buffer.add(feature_vec_queue[1:,:].cpu().numpy(), maneuver, act, rew, feature_vec_queue[:-1,:].cpu().numpy())
            ep_rew += rew
            steps += 1
            if steps >= BATCH_SIZE:
                sample = replay_buffer.sample()
                critic1_loss, critic2_loss = train_step_critic(*sample)
                c1_loss.append(critic1_loss.item())
                c2_loss.append(critic2_loss.item())
                if steps % POLICY_DELAY == 0:
                    actor_loss = train_step_action(*sample[:2])
                    a_loss.append(actor_loss.item())
            t.set_description(f"sz: {replay_buffer.rp_sz:5d} | steps/s: {steps / (time.perf_counter() - st):.2f} | rew/ep {avg_w(rews, 10):3.2f}| c1 loss: {avg_w(c1_loss):3.3f} | c2 loss: {avg_w(c2_loss):3.3f} | actor loss: {avg_w(a_loss):3.3f}")
            if terminated or truncated:
                break
        env.reset()
        rews.append(ep_rew)
        torch.save(actor.state_dict(), MODEL_SAVEFILE)
    print(f"Saved model to: {MODEL_SAVEFILE}")



    # Create the graphs with the appropriate suffix
    shift_suffix = "with_shift" if INCLUDE_SHIFT else "without_shift"
    create_critic_loss_graph(c1_loss, c2_loss,shift_suffix)
    create_action_loss_graph(a_loss,shift_suffix)
    create_ep_rew_graph(rews,shift_suffix)
    # Create the graphs with the appropriate filenames
    create_distance_graph(outer_dist, "outer",shift_suffix)
    create_distance_graph(dashed_dist, "dashed",shift_suffix)
    create_distance_graph(solid_dist, "solid",shift_suffix)
    create_cte_graph(ctes, "cte",shift_suffix)
    actor_weight_changes = calculate_weight_changes(actor_weight_history)
    critic1_weight_changes = calculate_weight_changes(critic1_weight_history)
    critic2_weight_changes = calculate_weight_changes(critic2_weight_history)
    # Plot weight changes for a specific layer
    plot_all_weight_changes("actor",actor_weight_changes, shift_suffix)
    plot_all_weight_changes("critic1",critic1_weight_changes,shift_suffix)
    plot_all_weight_changes("critic2",critic2_weight_changes,shift_suffix)

    # Assuming latent_space is a list of latent vectors
    latent_space_np = np.array(latent_space)

    # t-SNE Transformation
    tsne = TSNE(n_components=3, random_state=42)
    latent_space_3d = tsne.fit_transform(latent_space_np)

    # Plotting
    fig = px.scatter_3d(
        x=latent_space_3d[:, 0],
        y=latent_space_3d[:, 1],
        z=latent_space_3d[:, 2],
        title='3D t-SNE Plot of Latent Space',
        labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2', 'z': 't-SNE Component 3'}
    )
    fig.show()


    print("Evaluating:")
    tinycar_combo.actor = actor
    for maneuver in range(3):
        eval_dict = evaluate(tinycar_combo, env.unwrapped, maneuver=maneuver, steps=1000, episodes=5, render_mode=None, temporal=SEQ_LEN)
        cte_avg, cte_var, heading_error_avg, heading_error_var, terminations, steps_per_s, rew = eval_dict["cte_avg"], eval_dict["cte_var"], eval_dict["heading_error_avg"], eval_dict["heading_error_var"], eval_dict["terminations"], eval_dict["steps_per_s"], eval_dict["total_reward"] 
        print(f"Maneuver {maneuver} -> Total reward: {rew:.2f} | CTE: {cte_avg:.4f} m/step var: {cte_var:.4f}| Heading Error: {heading_error_avg:.4f} rad/step var {heading_error_var:.4f} | Terminations: {terminations:3d} | perf: {steps_per_s:.2f} steps/s")