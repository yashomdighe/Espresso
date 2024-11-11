import numpy as np
import yaml
from os.path import join
from pathlib import Path
import cv2
import open3d as o3d
import json
import pickle

from tqdm import tqdm
from PIL import Image

from pyrep.objects import VisionSensor

from rlbench.backend.const import *
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask
from rlbench.utils import get_stored_demos

ROOT = "/mnt/share/nas/Projects/Espresso/Data/RLBench_raw"
TASK = "slide_block_to_target"
VARIATION = 0
EPISODE = 0

if __name__ == "__main__":
    example_path = join(ROOT,TASK,f"variation{0}","episodes",f"episode{EPISODE}")
    print(example_path)
    with open(join(example_path, LOW_DIM_PICKLE), 'rb') as f:
        low_dim_obs = pickle.load(f)
    # low_dim_obs = pickle.load(low_dim_path)
    for i in range(len(low_dim_obs)):
        obs_dict = vars(low_dim_obs[i])
        for k, v in obs_dict.items():
            print(k)
        print(obs_dict['task_low_dim_state'])
        print(obs_dict['joint_positions'])
        print(obs_dict['gripper_pose'])
        break
        