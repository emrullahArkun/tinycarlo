from gymnasium import Wrapper
from typing import List, Union

class LanelineCrossingTerminationWrapper(Wrapper):
    def __init__(self, env, lanelines: Union[List[str], str]):
        """
        Wrapper class for terminating the environment when the car crosses certain lanelines.

        Args:
            env (gym.Env): The environment to wrap.
            lanelines (List[str] | str): List of laneline names or laneline name to check for crossing.
        """
        super().__init__(env)
        self.unwrapped.wrapped = True
        self.lanelines = lanelines if isinstance(lanelines, list) else [lanelines]

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        for layer_name in self.lanelines:
            if info["laneline_distances"][layer_name] <= self.unwrapped.car.track_width/2:
                terminated = True
        return observation, reward, terminated, truncated, info
    
class CTETerminationWrapper(Wrapper):
    def __init__(self, env, max_cte: float, number_of_steps: int = 1):
        """
        Wrapper class for terminating the environment based on the cross-track error (CTE) of the car to the lane path.

        Args:
            env (gym.Env): The environment to wrap.
            max_cte (float): The maximum acceptable CTE value in meters
        """
        super().__init__(env)
        self.unwrapped.wrapped = True
        self.max_cte = max_cte
        self.number_of_steps = number_of_steps
        self.steps_true = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if abs(info["cte"]) > self.max_cte:
            self.steps_true += 1
            if self.steps_true >= self.number_of_steps:
                terminated = True
                self.steps_true = 0
        else:
            self.steps_true = 0
        return observation, reward, terminated, truncated, info