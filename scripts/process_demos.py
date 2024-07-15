import numpy as np
import yaml
import os
from pathlib import Path
import cv2
from rlbench.backend.const import *
from tqdm import tqdm, trange

from PIL import Image
from natsort import natsorted
from pyrep.objects import VisionSensor
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import SlideBlockToTarget
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask
from rlbench.utils import get_stored_demos

from scipy.spatial.transform import Rotation as R
import open3d as o3d
from typing import Tuple


def make_colmap_camera_params(intrinsics: np.ndarray, 
                            extrinsics: np.ndarray, 
                            cam_id: int,
                            cam_name: str, 
                            img_size: int)-> Tuple[np.ndarray, np.ndarray]:
    colmap_int = np.array([cam_id, 
                           "PINHOLE",
                           img_size, img_size, 
                           intrinsics[0,0], intrinsics[1,1],
                           intrinsics[0,2], intrinsics[1,2]]) 
    r = R.from_matrix(extrinsics[:3, :3])
    q_vec = r.as_quat()
    t_vec = extrinsics[:3,3]
    colmap_ext = np.array([cam_id,
                           q_vec[3], q_vec[0], q_vec[1], q_vec[2],
                           t_vec[0], t_vec[1], t_vec[2],
                           cam_id,
                           f"{cam_name}.png"])

    return colmap_int, colmap_ext

def make_bg_transparent(img: np.ndarray) -> np.ndarray:
    alpha = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
    black_pixels = np.all(img == 0, axis=-1)
    alpha[black_pixels] = 0

    return np.dstack((img, alpha))

def filter_and_downsample_pcd(points: np.ndarray, colors: np.ndarray, Rx: np.ndarray, Ry: np.ndarray):
    distances = np.linalg.norm(points, axis=1)
    # points = np.dot(Ry[:3,:3], np.dot(Rx[:3,:3], points.T)).T
    indices = np.where((distances <= 2) # Filter by distance from origin
                       & (points[:, 1] > .5)  # Filter by height
                    # The two conditions below are needed for the setup for dynamic 3d gaussians where walls are super close
                    #    &((points[:, 2] < 0.8) & (points[:,2] > -1.1))  # Explicity Remove walls in the front and back
                    #    & ((points[:, 0] < 0.8) & (points[:,0] > -0.8)) # Explicitly Remove walls on the sides
                    ) 
    filtered_points = points[indices]
    filtered_colors = colors[indices]
    # filtered_points = points
    # filtered_colors = colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    return pcd.voxel_down_sample(voxel_size=0.01)

if __name__=="__main__":

    # Transforms from rlbench convention to colmap convention
    Rx = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    Ry = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])

    # Load Params
    root_dir = Path(__file__).resolve().parents[1]
    
    with open(Path.joinpath(root_dir, "params", "gaussian_splatting.yaml"), 'r') as f:
        cfg = yaml.safe_load(f)
    
    
    cameras= cfg["CAMERAS"]

    path_cfg = cfg["PATHS"]
    rlbench_data_path = Path.joinpath(Path(path_cfg["root"]), "RLBench_raw")
    
    dataset_cfg = cfg["DATASET"]
    img_size = dataset_cfg["img_size"]
    depth_in_m = dataset_cfg["depth_in_m"]
    masks_as_one_channel = dataset_cfg["masks_as_one_channel"]
    task = dataset_cfg["task"]
    num_demos = dataset_cfg["num_demos"]
    variations = dataset_cfg["variations"]
    mask_ids = dataset_cfg["mask_ids"]

    save_ply = cfg["SAVE_INIT_PLY"]

    camera_config = CameraConfig(image_size=(img_size, img_size),
                                 masks_as_one_channel=masks_as_one_channel,
                                 depth_in_meters=depth_in_m)
    
    obs_config = ObservationConfig(
        left_shoulder_camera=camera_config, 
        right_shoulder_camera=camera_config, 
        overhead_camera=camera_config,
        wrist_camera=camera_config, 
        front_camera=camera_config, 
        front_left_camera=camera_config, 
        front_right_camera=camera_config
    )
    obs_config.set_all(True)
    obs_cfg_dict = vars(obs_config)
    # print(f'Creating RLBench env...')

    # action_mode=MoveArmThenGripper(
    #     arm_action_mode=JointVelocity(), 
    #     gripper_action_mode=Discrete())

    # env = Environment(action_mode, str(rlbench_data_path), obs_config)
    # print(f"Launching environment")
    # env.launch()

    # print(f"Setting Task")
    # rlbench_task = env.get_task(SlideBlockToTarget)

    output_root = os.path.join(path_cfg["root"], "gaussian-splatting")

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print(f"Processing Task: {task}")
    task_output_path = os.path.join(output_root, task)
    if not os.path.exists(task_output_path):
        os.mkdir(task_output_path)
    
    for variation in variations:    
        print(f"Processing variation # {variation}")
        print(f"Getting RLBench saved demos...")

        demos = get_stored_demos(amount = num_demos, 
                                 dataset_root=rlbench_data_path,
                                 variation_number=variation,
                                 image_paths=True,
                                 task_name=task,
                                 obs_config=obs_config)
        print(f"Obtained RLBench saved demos")
        print(f"Processing demos")

        for episode_idx in trange(len(demos), desc="episode"):
            for timestep, demo in tqdm(enumerate(demos[episode_idx]), total=len(demos[episode_idx]), desc="timestep"):
                demo_dict = vars(demo)
                # for k, v in demo_dict.items():
                #     print(f"{k}: {type(v)}")
                # print(obs_dict["front_camera"].depth_noise)  
                # break
                timestep_path = os.path.join(task_output_path, f"variation_{variation}", f"episode_{episode_idx}", str(timestep))
                if not os.path.exists(timestep_path):
                    os.makedirs(timestep_path)
                
                points =[]
                colors = []
                intrinsics_li = []
                extrinsics_li = []

                im_path = os.path.join(timestep_path, "images")
                if not os.path.exists(im_path):
                    os.mkdir(im_path)

                sparse_path = os.path.join(timestep_path, "sparse", "0")
                if not os.path.exists(sparse_path):
                    os.makedirs(sparse_path)

                for cam_id, cam in cameras.items():

                    rgb = np.array(Image.open(demo_dict[f"{cam[0]}_rgb"]))
                    mask = rgb_handles_to_mask(np.array(Image.open(demo_dict[f"{cam[0]}_mask"])))                    # print(mask)
                    obj_ids = list(np.unique(mask))

                    misc_dict = demo_dict["misc"]
                    depth_img = image_to_float_array(
                        Image.open(demo_dict[f"{cam[0]}_depth"]),
                        DEPTH_SCALE
                    )
                    near = misc_dict[f"{cam[0]}_camera_near"]
                    far = misc_dict[f"{cam[0]}_camera_far"]
                    depth_in_m = near + depth_img * (far - near)
                    depth_in_m = obs_cfg_dict[f"{cam[0]}_camera"].depth_noise.apply(depth_in_m)
                    
                    cam_ext = Ry@Rx@misc_dict[f"{cam[0]}_camera_extrinsics"]
                    points.append(
                        VisionSensor.pointcloud_from_depth_and_camera_params(
                            depth_in_m,
                            cam_ext,
                            misc_dict[f"{cam[0]}_camera_intrinsics"]
                        ).reshape(img_size*img_size,3)
                    )
            
                    colors.append(rgb.reshape(img_size*img_size,3)/255)

                    rgb = make_bg_transparent(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(im_path,f"{cam[1]}.png"), rgb)

                    mask_img = 1 - np.isin(mask, mask_ids).astype(np.uint8)
                    cv2.imwrite(os.path.join(im_path,f"{cam[1]}_mask.png"), mask_img*255)
                    
                    intrinsics, extrinsics = make_colmap_camera_params(
                        misc_dict[f"{cam[0]}_camera_intrinsics"],
                        cam_ext.T,
                        cam_id,
                        cam[1],
                        img_size
                    )
                    intrinsics_li.append(intrinsics)
                    extrinsics_li.append(extrinsics)
                # break
                intrinsics = np.vstack(intrinsics_li)
                extrinsics = np.vstack(extrinsics_li)

                pcd = filter_and_downsample_pcd(np.vstack(points), np.vstack(colors), Rx, Ry)
                merged_points = np.asarray(pcd.points)
                merged_colors = np.asarray(pcd.colors)*255 
                id = np.linspace(0, merged_points.shape[0]-1, merged_points.shape[0], dtype=int).reshape(-1,1)
                err = -1*np.ones((merged_points.shape[0], 1), dtype=int)

                np.savetxt(os.path.join(sparse_path,"points3D.txt"), 
                           np.hstack([id[::-1], merged_points, merged_colors.astype(int), err]),
                           fmt='%d %f %f %f %d %d %d %d ')
                
                np.savetxt(os.path.join(sparse_path,"cameras.txt"),
                           intrinsics,
                           fmt="%s %s %s %s %s %s %s %s")
                
                np.savetxt(os.path.join(sparse_path,"images.txt"),
                           extrinsics,
                           fmt="%s %s %s %s %s %s %s %s %s %s\n")
                
                if save_ply & (timestep == 0):
                    print(f"\nSaving initial pcd as ply")
                    o3d.io.write_point_cloud(os.path.join(task_output_path, f"variation_{variation}", f"episode_{episode_idx}", "init_pcd.ply"), pcd)
        #         break
        #     break
        # break
    print("DONE")