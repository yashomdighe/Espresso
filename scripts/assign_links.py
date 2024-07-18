import numpy as np
import open3d as o3d
import os
from pathlib import Path
from typing import Tuple

ROOT = '/Volumes/dl-primary/Projects/Espresso/Data/dynamic_3dgs'

def get_gaussian_centers(params: dict,  timestep: int) -> Tuple[np.ndarray, np.ndarray]:
    
    points = np.ascontiguousarray(params["means3D"][timestep]).astype(float)
    colors = np.ascontiguousarray(params["rgb_colors"][timestep]).astype(float)

    return points, colors

if __name__=='__main__':
    # print(os.getcwd())
    exp = "test_15"
    seq = "slide_block_to_target/episode_0"
    # output_path = os.path.join(ROOT, "outputs")
    params = dict(np.load(os.path.join(os.getcwd(), "data", str(exp), str(seq), "params.npz")))

    t0_points, t0_colors = get_gaussian_centers(params, 0)
    t1_points, t1_colors = get_gaussian_centers(params, -1)

    del params

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(t0_points)
    pcd0.colors = o3d.utility.Vector3dVector(t0_colors)
    # o3d.io.write_point_cloud("t_0.ply", pcd0)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(t1_points)
    pcd1.colors = o3d.utility.Vector3dVector(t1_colors)
    # o3d.io.write_point_cloud("t_final.ply", pcd1)
    o3d.visualization.draw_geometries([pcd0, pcd1])