import numpy as np
import os
import open3d as o3d
import copy

import yaml
from typing import Tuple

from robofin import samplers
from robofin.robot_constants import FrankaConstants

def get_gaussian_centers(params: dict,  timestep: int) -> Tuple[np.ndarray, np.ndarray]:
    
    points = np.ascontiguousarray(params["means3D"][timestep]).astype(float)
    colors = np.ascontiguousarray(params["rgb_colors"][timestep]).astype(float)
    seg = np.ascontiguousarray(params['seg_colors']).astype(float)

    return points, colors, seg

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    # o3d.visualization.draw_geometries([source_temp, target_temp])

def get_franka_pcd(joint_config: np.ndarray ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    
    sampler = samplers.NumpyFrankaSampler(
        num_robot_points=16384, num_eef_points=512, use_cache=True, with_base_link=True
    )
    samples = sampler.sample(joint_config,0.04)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(samples[:,:3]).astype(float))

    links = links = samples[:,3][...,None]

    return pcd, links

ROOT = '/mnt/share/nas/Projects/Espresso/Data/dynamic_3dgs'

if __name__=="__main__":
    exp = "test_15"
    seq = "slide_block_to_target/episode_0"
    params = dict(np.load(os.path.join(ROOT, "output", str(seq), str(exp), "params.npz")))

    init_joint_positions = [4.29511238e-06,  1.75092965e-01,  8.85453846e-06, -8.73115242e-01, -2.25986332e-06, 1.22154474e+00, 7.85391450e-01]
    t_vec = np.array([-2.981e-01, -1.27797e-02, +7.2005e-01])
    t = np.vstack([np.hstack([np.eye(3), t_vec.reshape(-1,1)]), np.array([0, 0, 0, 1])])

    Rx = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    Ry = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
    transform_init = np.ascontiguousarray(Ry@Rx@t).astype(float)
    # print(transform_init)

    src_pcd, src_links = get_franka_pcd(np.array(init_joint_positions))
    t0_points, t0_colors, t0_seg = get_gaussian_centers(params, 0)
 
    del params, t0_colors
    indices0 = (t0_seg == np.array([1., 0., 0,])).all(axis=1)
    filtered_points0 = t0_points[indices0]
    indices1 = np.where(filtered_points0[:, 2] > -0.3)
    filtered_points = filtered_points0[indices1]

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(filtered_points))

    # draw_registration_result(src_pcd, target_pcd, transform_init)
    threshold = 0.02
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        src_pcd, target_pcd, threshold, transform_init
        )
    print(evaluation)
    print("Apply point-to-point ICP")
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        src_pcd, target_pcd, threshold, transform_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)

    src_pcd.transform(reg_p2p.transformation)

    src_tree = o3d.geometry.KDTreeFlann(src_pcd)

    links = []
    for point in filtered_points:
        [_, idx, _] = src_tree.search_knn_vector_3d(point, 20)
        
        # get link of the nearest neighbours
        neighbour_links = src_links[(np.asarray(idx))]

        # get count of each link
        link, counts = np.unique(src_links[(np.asarray(idx))], return_counts=True)

        # assign the link with the highest count to the point
        links.append(link[np.argmax(counts)])
        # break
    
    # assign the color based on the links using a colormap
    with open("link_color_map.yaml", "r") as f:
        color_map = yaml.safe_load(f)
    colors = []
    
    for link in links:
        # print(link[0])
        colors.append(color_map[(link)])
    # print(colors)

    target_pcd.colors = o3d.utility.Vector3dVector((np.ascontiguousarray(colors)/255).astype(float))
    o3d.io.write_point_cloud("link_colored.ply", target_pcd)

    


