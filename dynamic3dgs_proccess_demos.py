import numpy as np
import yaml
import os
from pathlib import Path
import cv2
import open3d as o3d
import json

from tqdm import tqdm
from PIL import Image

from pyrep.objects import VisionSensor

from rlbench.backend.const import *
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask
from rlbench.utils import get_stored_demos

if __name__=="__main__":

    # Transforms from rlbench convention to colmap convention
    Rx = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    Ry = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])

    # Load Params
    root_dir = Path(__file__).resolve().parents[1]
    
    with open(Path.joinpath(root_dir, "params", "dynamic_3dgs.yaml"), 'r') as f:
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

    # save_ply = cfg["SAVE_INIT_PLY"]

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

    output_root = os.path.join(path_cfg["root"], "dynamic_3dgs")

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
                                 obs_config=obs_config,
                                 random_selection=False)
        print(f"Obtained RLBench saved demos")
        print(f"Processing demos")

        for episode_idx in range(len(demos)):

            # episode_path = os.path.join(task_output_path, f"variation_{variation}", f"episode_{episode_idx}")
            episode_save_path = os.path.join(task_output_path, f"episode_{episode_idx}")
            for cam_id, cam in cameras.items():
                im_path = os.path.join(episode_save_path, "ims", str(cam[1]))
                seg_path = os.path.join(episode_save_path, "seg", str(cam[1]))

                if not os.path.exists(im_path):
                    os.makedirs(im_path)
                
                if not os.path.exists(seg_path):
                    os.makedirs(seg_path)
            k_epi_li = []
            w2c_epi_li = []
            fn_epi_li = []
            cam_id_epi_li = []

            for timestep, demo in tqdm(enumerate(demos[episode_idx]), total=len(demos[episode_idx]), desc=f"Episode {episode_idx+1}/{len(demos)}"):
                demo_dict = vars(demo)
                # for k, v in demo_dict.items():
                #     print(f"{k}: {type(v)}")
                # print(obs_dict["front_camera"].depth_noise)  
                # break
                # timestep_path = os.path.join(task_output_path, f"variation_{variation}", f"episode_{episode_idx}", str(timestep))
                # if not os.path.exists(timestep_path):
                #     os.makedirs(timestep_path)
                
                points =[]
                colors = []
                pcd_seg = []
                k_timestep_li = []
                w2c_timestep_li = []
                fn_timestep_li = []
                cam_id_timestep_li = []
                episode_im_path = os.path.join(episode_save_path, "ims")
                
                episode_seg_path = os.path.join(episode_save_path, "seg")

                for cam_id, cam in cameras.items():
                    
                    # Process vision data
                    bgr = cv2.cvtColor(np.array(Image.open(demo_dict[f"{cam[0]}_rgb"])), cv2.COLOR_RGB2BGR)
                    mask = rgb_handles_to_mask(np.array(Image.open(demo_dict[f"{cam[0]}_mask"])))                    # print(mask)

                    misc_dict = demo_dict["misc"]
                    cam_ext = Ry@Rx@misc_dict[f"{cam[0]}_camera_extrinsics"]

                    
                    # # Debugging to figure out mask ids    
                    # obj_ids = list(np.unique(mask))
                    # print(obj_ids)
                    # for o_id, obj_id in enumerate(obj_ids):
                    #     masked_image = (mask == obj_id)[..., None] * bgr
                    #     os.makedirs(f'{cam[0]}/', exist_ok=True)
                    #     cv2.imwrite(f'{cam[0]}/{obj_id}.png', masked_image)

                    # Save rgb
                    cv2.imwrite(os.path.join(episode_im_path, str(cam[1]), "{:06d}.png".format(timestep)), bgr)
                    # Save mask
                    mask_img = 1 - np.isin(mask, mask_ids).astype(np.uint8)
                    cv2.imwrite(os.path.join(episode_seg_path, str(cam[1]),"{:06d}.png".format(timestep)), mask_img*255)
                    
                    
                    if timestep == 0:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        depth_img = image_to_float_array(
                            Image.open(demo_dict[f"{cam[0]}_depth"]),
                            DEPTH_SCALE
                        )
                        near = misc_dict[f"{cam[0]}_camera_near"]
                        far = misc_dict[f"{cam[0]}_camera_far"]
                        depth_in_m = near + depth_img * (far - near)
                        depth_in_m = obs_cfg_dict[f"{cam[0]}_camera"].depth_noise.apply(depth_in_m)
                        
                        points.append(
                            VisionSensor.pointcloud_from_depth_and_camera_params(
                                depth_in_m,
                                cam_ext,
                                misc_dict[f"{cam[0]}_camera_intrinsics"]
                            ).reshape(img_size*img_size,3)
                        )
                        colors.append(rgb.reshape(img_size*img_size,3)/255)
                        pcd_seg.append(mask_img.reshape(img_size*img_size,1)/255)
                    
                    # Append cam_id, fn, k, w2c to the current timestep lists
                    cam_id_timestep_li.append(cam[1])
                    fn_timestep_li.append(os.path.join(str(cam[1]), "{:06d}.png".format(timestep)))
                    k_timestep_li.append(misc_dict[f"{cam[0]}_camera_intrinsics"].tolist())
                    w2c_timestep_li.append(np.linalg.inv(cam_ext).tolist())

                # break
                if timestep == 0:
                    points = np.vstack(points)
                    colors = np.vstack(colors)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.01)
                    # pcd = filter_and_downsample_pcd(np.vstack(points), np.vstack(colors))
                    filtered_points = np.ascontiguousarray(downsampled_pcd.points)
                    filtered_colors = np.ascontiguousarray(downsampled_pcd.colors)
                    pcd_seg = np.zeros((filtered_points.shape[0],1), dtype=np.float64)
                    distances = np.linalg.norm(filtered_points, axis=1)
                    indices = np.where((distances <= 2) & # Filter by distance from origin
                                    (filtered_points[:, 1] > .752) & # Filter by height
                                    ((filtered_points[:, 2] < 0.8) & (filtered_points[:,2] > -1.1)) & # Remove walls in the front and back
                                    ((filtered_points[:, 0] < 0.8) & (filtered_points[:,0] > -0.8))) # Remove walls on the sides
                    pcd_seg[indices] = 1
                    pcd_array = np.hstack((filtered_points, filtered_colors, pcd_seg))
                    tqdm.write(f"Saving initial pointcloud with {pcd_array.shape[0]} points")
                    np.savez(os.path.join(episode_save_path, "init_pt_cld.npz"), data=pcd_array)

                    # Free memory
                    del pcd, downsampled_pcd, 

                k_epi_li.append(k_timestep_li)
                w2c_epi_li.append(w2c_timestep_li)
                fn_epi_li.append(fn_timestep_li)
                cam_id_epi_li.append(cam_id_timestep_li)
            # break
            meta_data = {
                "w": img_size,
                "h": img_size,
                "k": k_epi_li,
                "w2c": w2c_epi_li,
                "fn": fn_epi_li,
                "cam_id": cam_id_epi_li
            }
            with open(os.path.join(episode_save_path, "train_meta.json"), "w") as outfile: 
                json.dump(meta_data, outfile)    
                # if save_ply & (timestep == 0):
                #     tqdm.write(f"Saving initial pcd for episode {episode_idx}")
                #     o3d.io.write_point_cloud(os.path.join(task_output_path, f"variation_{variation}", f"episode_{episode_idx}", "init_pcd.ply"), pcd)
            
        # break
    print("DONE")