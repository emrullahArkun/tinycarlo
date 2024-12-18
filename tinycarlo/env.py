import cv2
import numpy as np
import time
import yaml
import os
from typing import Dict, Optional, Tuple, Union, Any
import gymnasium as gym

from tinycarlo.renderer import Renderer
from tinycarlo.car import Car
from tinycarlo.map import Map
from tinycarlo.camera import Camera
from tinycarlo.helper import getenv

class TinyCarloEnv(gym.Env):
    metadata: Dict[str, list] = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None, config: Optional[Union[str, Dict[str, Any]]] = None):
        """
        config can be provided as either path to yaml file or as dictionary.
        render_mode can be either "human" or "rgb_array".

        human render mode will open a window and display the map and camera view.
        rgb_array render mode will return the camera view as numpy array in rgb format (independent from observation mode).
        """
        self.config_path: Optional[str] = None
        if isinstance(config, str):
            if config.endswith(".yaml"):
                self.config_path = os.path.abspath(config)
            else:
                self.config_path = os.path.abspath(os.path.join(config, "config.yaml"))
            with open(self.config_path, "r") as stream:
                config = yaml.safe_load(stream)
                print(f'Loaded configuration file: {self.config_path}')
        self.config = config

        """
        Setting up variables needed for simulation including car, map, camera, reward handler, etc.
        """
        self.fps: int = config['sim'].get('fps', 30)
        self.T: float = 1/self.fps
        self.render_realtime: bool = config['sim'].get('render_realtime', False)
        self.observation_space_format: str = config['sim'].get('observation_space_format', "rgb")
        self.overview_pixel_per_meter: int = config['sim'].get('overview_pixel_per_meter', 150)
        self.render_node_names: bool = config['sim'].get('render_node_names', False)

        self.map: Map = Map(config['map'], base_path=self.config_path)
        self.car: Car = Car(self.T, self.map, config['car'])

        self.renderer: Renderer = Renderer(self.map, self.car, self.overview_pixel_per_meter)
        self.camera: Camera = Camera(self.map, self.car, self.renderer, config['camera'])
        self.loop_time: int = 1
        self.window: Optional[str] = None
        self.window_camera: Optional[str] = None

        self.wrapped: bool = False # flag to check if env is wrapped by a custom wrapper. I true, it will disable default reward and termination condition.

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode: Optional[str] = render_mode
        self.no_observation: bool = False # NOTE: with this flag the observation rendering can be disabled if render_mode is also None

        """
        Gym specific setup of action and observation space
        """
        # action space: {"car_control": [velocity, steering_angle], "maneuver": discrete maneuver (straight, right, u-turn, left)}
        self.action_space: gym.spaces.Dict = gym.spaces.Dict({"car_control": gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32), "maneuver": gym.spaces.Discrete(4)})
        # observation space: camera views
        if self.observation_space_format == "rgb":
            observation_space_shape: Tuple[int, int, int] = self.camera.resolution + [3]
        else:
            n_classes = len(self.map.get_laneline_names())
            observation_space_shape: Tuple[int, int, int] = [n_classes] + self.camera.resolution
        self.observation_space: gym.spaces.Box = gym.spaces.Box(low=0, high=255, shape=observation_space_shape, dtype=np.uint8)

        self.reset()

    def __get_obs(self) -> np.ndarray:
        if not self.no_observation or self.render_mode is not None:
            return self.camera.capture_frame(self.observation_space_format)
        else:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
    
    def __get_info(self) -> Dict[str, Any]:
        cte, heading_error, distances, local_path, velocity = self.car.get_info()
        return {"cte": cte, "heading_error": heading_error, "position": self.car.position.copy(), "orientation": self.car.rotation, "laneline_distances": distances, "local_path": local_path, "velocity": velocity}
    
    def __default_reward(self, cte: float) -> float: 
        """
        Calculates a default reward solely based on cte with a linear shaping.

        Car track width is used as reference for the shaping. linear function reaches 0 when cte == track_width
        """
        return max((-1/self.car.track_width) * cte + 1, 0)

    def __default_termination(self, cte: float) -> bool:
        """
        Default termination condition based on cte.
        """
        return cte > (self.car.track_width * 10)

    def reset(self, seed: Optional[int] = None, options: Optional[Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # reset car position to random spawn point
        self.car.reset(self.np_random)

        observation: np.ndarray = self.__get_obs()
        info: Dict[str, Any] = self.__get_info()

        if self.render_mode == "human":
            self.__render_frame()

        return observation, info

    def step(self, action: Union[gym.spaces.Dict, Dict[str, Any]]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        st: float = time.perf_counter()
        # clip car_control to action space
        car_control: np.ndarray = np.clip(action["car_control"], self.action_space["car_control"].low, self.action_space["car_control"].high)
        st_step: float = time.perf_counter()
        car_truncated = self.car.step(car_control[0], car_control[1], action["maneuver"])
        td_step: float = time.perf_counter() - st_step

        st_obs: float = time.perf_counter()
        observation: np.ndarray = self.__get_obs()
        td_obs: float = time.perf_counter() - st_obs

        st_info: float = time.perf_counter()
        info: Dict[str: Any] = self.__get_info()
        td_info: float = time.perf_counter() - st_info

        """
        This is the default reward and termination condition.
        To change reward and termination, use tinycarlo.wrappers or define a custom wrapper.
        info can be useful to calculate more complex rewards or termination conditions.
        """
        cte = info["cte"]
        reward: float = self.__default_reward(cte) if self.wrapped == False else 0
        terminated: bool = self.__default_termination(cte) if self.wrapped == False else False

        if self.render_mode == "human":
            self.__render_frame()

        # for debugging performance
        if getenv("DEBUG"):
            print(f"all: {(time.perf_counter() - st)*1000:.2f} ms | obs render {td_obs*1000:.2f} ms | info {td_info*1000:.2f} ms | car step {td_step*1000:.2f} ms")

        return observation, reward, terminated, car_truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self.__render_frame()

    def __render_frame(self) -> Optional[np.ndarray]:
        if self.window is None and self.render_mode == "human":
            self.window = "Map"
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        if self.window_camera is None and self.render_mode == "human":
            self.window_camera = "Camera"
            cv2.namedWindow(self.window_camera, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        
        camera_view: np.ndarray = self.camera.get_last_frame_rgb()

        if self.render_mode == "human":
            overview: np.ndarray = self.renderer.render_overview()

            cv2.imshow(self.window, overview)
            cv2.imshow(self.window_camera, camera_view)
        
            waiting_time: float = self.T - self.loop_time
            if waiting_time < 0.001 or self.render_realtime == False:
                waiting_time = 0.001
            cv2.waitKey(int(waiting_time*1000))
        else:
            return camera_view
        
    def close(self) -> None:
        if self.window is not None:
            cv2.destroyWindow(self.window)
        if self.window_camera is not None:
            cv2.destroyWindow(self.window_camera)