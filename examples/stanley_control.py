import gymnasium as gym
import tinycarlo
import os
import math

from tinycarlo.wrapper.reward import CTESparseRewardWrapper
from tinycarlo.wrapper.termination import LanelineCrossingTerminationWrapper, CTETerminationWrapper

config = {
    "sim": {
        "fps": 30,
        "render_realtime": True,
        "observation_space_format": "rgb", # "rgb" or "classes"
        "real_world_env": "autosys"
    },
    "car": {
        "wheelbase": 0.065,#0.0487, # distance between front and rear axle in meters
        "track_width": 0.027, # distance between the left and right wheel in meters
        "max_velocity": 0.15, # in m/s
        "max_steering_angle": 30, # in degrees
        "steering_speed": 30, # in deg/s
        "max_acceleration": 0.1, # in m/s^2
        "max_deceleration": 1, # in m/s^2
        "tinycar_hostname": "192.168.84.49",
    },
    "camera": {
        "position": [0.02, 0, 0.024], # [x,y,z] in m relative to middle of front axle (x: forward, y: right, z: up)
        "orientation": [15, 0, 0], # [pitch,roll,yaw] in degrees
        "resolution": [240, 320], # [height, width] in pixels
        "fov": 120, # in degrees
        "max_range": 0.5, # in meters
        "line_thickness": 6 # in pixels
    },
    "map": {
        "json_path": os.path.join(os.path.dirname(__file__), "maps/simple_layout.json"),
        "pixel_per_meter": 450 # 222
    }
}

env = gym.make("tinycarlo-realworld-v2", config=config, render_mode="human")
env = CTESparseRewardWrapper(env, 0.01)
env = CTETerminationWrapper(env, 0.07, number_of_steps=5)

k = 4
speed = 0.4

observation, info = env.reset(seed=2)

import time

while True:
    st = time.perf_counter()
    cte, heading_error = info["cte"], info["heading_error"]
    # Lateral Control with Stanley Controller
    steering_correction = math.atan2(k * cte, speed)
    steering_angle = (heading_error + steering_correction) * 180 / math.pi / config["car"]["max_steering_angle"]
    action = {"car_control": [speed, steering_angle], "maneuver": 3} # always try to turn left
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
        print("Resetting the environment.")
    print(f"steps/s: {1/(time.perf_counter()-st):.2f}", end="\r")

env.close()