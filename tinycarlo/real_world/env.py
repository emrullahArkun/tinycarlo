from typing import Dict, Optional, Tuple, Union, Any
import importlib, inspect, functools

from tinycarlo.env import TinyCarloEnv
from tinycarlo.camera import Camera
from tinycarlo.car import Car
from tinycarlo.renderer import Renderer

class TinyCarloRealWorldEnv(TinyCarloEnv):

    def __get_env_modules(self, env_name: str):
        return [cls for _, cls in inspect.getmembers(importlib.import_module(f"tinycarlo.real_world.environments.env_{env_name.lower()}")) if inspect.isclass(cls)]
    def __get_camera_dyn(self, modules):
        return [cam for cam in modules if issubclass(cam, Camera) and cam.__name__ != "Camera"][0]
    def __get_car_dyn(self, modules):
        return [car for car in modules if issubclass(car, Car) and car.__name__ != "Car"][0]

    def __init__(self, render_mode: Optional[str] = None, config: Optional[Union[str, Dict[str, Any]]] = None):
        super().__init__(render_mode, config)
        self.real_world_env_name = self.config["sim"].get("real_world_env", None)
        if self.real_world_env_name is None:
            raise ValueError("Real world environment is not provided in config. Use normal tinycarlo env instead.")
        modules = self.__get_env_modules(self.real_world_env_name)
        self.car = self.__get_car_dyn(modules)(self.T, self.map, self.config["car"])
        self.renderer = Renderer(self.map, self.car, self.overview_pixel_per_meter)
        self.camera = self.__get_camera_dyn(modules)(self.map, self.car, self.renderer, self.config["camera"])
        self.reset()
        


