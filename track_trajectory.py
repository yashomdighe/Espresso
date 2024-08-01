import meshcat_shapes
import numpy as np
import qpsolvers
from loop_rate_limiters import RateLimiter
import time

import pink
from pink import solve_ik
import pinocchio as pin # type: ignore
from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector
from pink.visualization import start_meshcat_visualizer

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try ``pip install robot_descriptions``"
    ) from exc

if __name__ == "__main__":
    robot = load_robot_description("panda_description")
    model = robot.model
    data = robot.data
    q_ref = np.array(
        [4.29511238e-06,  
         1.75092965e-01,  
         8.85453846e-06, 
         -8.73115242e-01, 
         -2.25986332e-06, 
         1.22154474e+00, 
         7.85391450e-01,
        0.02542847,
        0.02869188,]
    )
    pin.forwardKinematics(model, data, q_ref)
    pin.updateFramePlacements(model, data)