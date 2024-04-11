# Real World Tinycarlo
Here you will find the codebase for the tinycarlo real-world environment. Instead of testing and training solely in a simulation, you can use this env to test and train in the real-world. Implement your own hardware interface and use the provided code and you have a fully working digital twin of your environment.

## Create your own environment
To create your own environment, you add a env_ file inside the environments folder. This file should contain at least two classes:
- Custom Camera (inherits from tinycarlo.camera):
- Custom Car (inherits from tinycarlo.car): You should override the `reset()` and `step()` methods. Inside the `step()` method you must set `position` and `rotation` of the car. 

**Note:** As an example, you can look at the `env_autosys.py` file. This environment uses tracking information from camera modules, which come in as UDP messages on a multicast address. The car is controlled with an external library. The Camera also receives its raw images from that library. However, a neural network (also external library) processes these images to create tinycarlo like observation images.