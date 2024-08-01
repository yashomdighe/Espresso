import meshcat_shapes
import numpy as np
import qpsolvers
from loop_rate_limiters import RateLimiter
import time
import os
from typing import Tuple
import pink
from pink import solve_ik
import pinocchio as pin # type: ignore
from pink.tasks import FrameTask
from pink.utils import custom_configuration_vector
from pink.visualization import start_meshcat_visualizer

from scripts.assign_links import assign_links, get_gaussian_centers

rng = np.random.default_rng(0)

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try ``pip install robot_descriptions``"
    ) from exc

def choose_frames_centers(points, links, indices, num_pts: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    # Sample X points per link from the gs
    parents = []
    centers = []
    idx = []
    
    for link in np.unique(links):
        # print(link)
        id = np.where((links == link))
        # print(id[0])
        sample_idx = rng.choice(id[0], 1)[0]
        # print(f"link: {link}, center: {points[sample_idx[0]]}, idx: {sample_idx[0]}")
        # break
        centers.append(points[sample_idx])
        idx.append(indices[sample_idx][0])
        parents.append(link)

    return np.hstack((np.array(centers), np.array(parents)[...,None])), np.array(idx)


def compute_tranform_3dgs_to_pin(tf) -> np.ndarray:
    
    # t_vec = np.array([-2.981e-01, -1.27797e-02, +7.2005e-01])
    # t = np.vstack([np.hstack([np.eye(3), t_vec.reshape(-1,1)]), np.array([0, 0, 0, 1])])
    # Rx = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    # Ry = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
    # transform = Ry@Rx@t
    R = tf[:3,:3].T
    T= -1*R@tf[:3,3]
    X = np.vstack((np.hstack((R,T[:3].reshape(-1,1))), np.array([0,0,0,1])))

    return X

def set_task_targets():
    pass

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


    # TO DO: Do this using os.path
    read_from_file = True

    if read_from_file:
        data = np.loadtxt("sampled_pts.txt")
        points = data[:,:3]
        links = data[:,3][...,None]
        indices = data[:,4][...,None]
    else: 
        # Load and get links and pts from the gs
        ROOT = '/mnt/share/nas/Projects/Espresso/Data/dynamic_3dgs'
        exp = "test_15"
        seq = "slide_block_to_target/episode_0"
        path = os.path.join(ROOT, "output", str(seq), str(exp))
        points, links, indices = assign_links(path, True)
    

    # print(indices.shape)
    # Randomly? Choose N points per link in the gs
    centers, idx = choose_frames_centers(points, links, indices)

    transfrom_3dgs_to_pin = compute_tranform_3dgs_to_pin(transform)

    skip = [0.0, 9.0, 10.0]
    frames = {}
    # Create frames from the centers
    for center in centers:
        link = center[3:][0]
        if link in skip:
            continue
        coords = transfrom_3dgs_to_pin@np.append(center[:3], 1).reshape(-1,1)
        tf = np.hstack([np.vstack([np.eye(3), np.zeros((1,3))]), coords])
        frames[f"pt{int(link)}"] = tf
        # print(coords.T, link)

    viz = start_meshcat_visualizer(robot)

    viewer = viz.viewer
    for k, v in frames.items():
        # print(f"{k}:{v}")
        meshcat_shapes.frame(viewer[k], opacity=0.5)
        viewer[k].set_transform(v)



        # Create and add frame to body
        # Create task and add to list
    viz.display(q_ref)
    while True: 
        pass
        

    # Set task targets
    # Solve IK


    







