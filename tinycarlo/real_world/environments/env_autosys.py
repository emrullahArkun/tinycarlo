###### Environment for the autosys research lab environment ######
# https://autosys-lab.de/platforms/2020h0streetplatform/
from typing import Dict, Optional, Tuple, Union, Any
import time, math, socket, struct, threading
import numpy as np

from tinycarlo.camera import Camera
from tinycarlo.car import Car
from tinycarlo.helper import clip_angle

from tinycar import Tinycar, TinycarTelemetry

class AutosysCamera(Camera):
    def __init__(self, map, car, renderer, config):
        super().__init__(map, car, renderer, config)
        if isinstance(self.car, AutosysCar):
            self.tinycar = self.car.tinycar
        else:
            raise ValueError("Car must be an instance of AutosysCar")
    
    def capture_frame(self, format: str) -> np.ndarray:
        return np.zeros((self.resolution[0], self.resolution[1], 3), dtype=np.uint8)

class CarTracking():
    MCAST_GRP = '239.255.255.250'
    MCAST_PORT = 5565
     
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.MCAST_GRP, self.MCAST_PORT))
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 32) 
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(self.MCAST_GRP) + socket.inet_aton('0.0.0.0'))
        self.message_format = 'BHHf'
        self.receive_th = threading.Thread(target=self.receive)
        self.running = False
        self.id_2_transform = np.eye(3)
        self.last_tracking_data = None

    def transform_tracking_data(self, id, x,y, ori):
        if id == 0:
            return None
        if id == 1:
            return (x, y, ori)
        if id == 2:
            pos_vec = self.id_2_transform.dot(np.array([x, y, 1]))
            return (pos_vec[0], pos_vec[1], ori)
        
    def get_tracking_data(self):
        tracking_data = self.last_tracking_data
        self.last_tracking_data = None
        return tracking_data
        
    def start(self):
        self.running = True
        self.receive_th.start()
    
    def stop(self):
        self.running = False
        self.receive_th.join()
    
    def receive(self):
        while self.running:
            data, _ = self.sock.recvfrom(12)
            id, position_x, position_y, orientation = struct.unpack(self.message_format, data)
            self.last_tracking_data = self.transform_tracking_data(id, position_x, position_y, orientation)

class AutosysCar(Car):
    def __init__(self, T, map, config):
        super().__init__(T, map, config)
        self.tinycar_hostname = config.get('tinycar_hostname', 'localhost')
        self.tinycar = Tinycar(self.tinycar_hostname)
        self.position_check_thres = 4
        self.reset_speed = 0.3
        self.tracking = CarTracking()
        self.tracking.start()

    def step(self, velocity: float, steering_angle: float, maneuver: int) -> bool:
        # steering angle in tinycarlo is -1 to 1, but for tinycar it is in degrees
        steering_angle = steering_angle * self.max_steering_angle
        self.drive(steering_angle, velocity)
        self.radius = self.wheelbase / (math.tan(math.radians(steering_angle)))
        while (tracking_data := self.tracking.get_tracking_data()) is None:
            pass
        self.position = tracking_data[:2]
        self.rotation = tracking_data[2]
        self.__update_position_front()

        return self.find_local_path(maneuver=maneuver)
    
    def reset(self, np_random: Any) -> None:
        """
        Resets the car by selecting the nearest edge from position and driving to the nearest node.
        """
        desired_position, desired_rotation, nearest_edge = self.map.sample_nearest_edge(self.position)
        while not self.check_position(desired_position):
            # Use stanley controller to drive to the nearest node
            cte = self.map.lanepath.distance_to_edge(self.position, nearest_edge)
            heading_error = clip_angle(desired_rotation - self.rotation)
            steering_correction = math.atan2(5 * cte, self.reset_speed)
            steering_angle = (heading_error + steering_correction) * 180 / math.pi
            self.drive(steering_angle, self.reset_speed)
        # arrived nearly at the desired position
        self.local_path = [nearest_edge]
        self.steering_angle = 0.0
        self.velocity = 0.0
        self.last_maneuver = 0
    
    def drive(self, steering_angle, speed):
        self.tinycar.setServoAngle(9000 + int(steering_angle * 100))
        self.tinycar.setMotorDutyCycle(speed * 1000)
        time.sleep(self.T)

    def check_position(self, desired_position):
        return self.position[0] >= desired_position[0] - self.position_check_thres and self.position[0] <= desired_position[0] + self.position_check_thres and self.position[1] >= desired_position[1] - self.position_check_thres and self.position[1] <= desired_position[1] + self.position_check_thres




