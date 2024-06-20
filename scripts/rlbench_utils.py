import os
import open3d as o3d
import numpy as np
from pyrep.const import RenderMode
from rlbench.utils import get_stored_demos
from rlbench.observation_config import ObservationConfig

DATASET_ROOT=os.environ['RLBENCH_DATA']

obs_confing = ObservationConfig()

img_size = [512,512]

obs_config = ObservationConfig()
obs_config.set_all(True)
obs_config.right_shoulder_camera.image_size = img_size
obs_config.left_shoulder_camera.image_size = img_size
obs_config.overhead_camera.image_size = img_size
obs_config.wrist_camera.image_size = img_size
obs_config.front_camera.image_size = img_size
obs_config.front_left_camera.image_size = img_size
obs_config.front_right_camera.image_size = img_size

# Store depth as 0 - 1
obs_config.right_shoulder_camera.depth_in_meters = False
obs_config.left_shoulder_camera.depth_in_meters = False
obs_config.overhead_camera.depth_in_meters = False
obs_config.wrist_camera.depth_in_meters = False
obs_config.front_camera.depth_in_meters = False
obs_config.front_left_camera.depth_in_meters = False
obs_config.front_right_camera.depth_in_meters = False

# We want to save the masks as rgb encodings.
obs_config.left_shoulder_camera.masks_as_one_channel = False
obs_config.right_shoulder_camera.masks_as_one_channel = False
obs_config.overhead_camera.masks_as_one_channel = False
obs_config.wrist_camera.masks_as_one_channel = False
obs_config.front_camera.masks_as_one_channel = False
obs_config.front_left_camera.masks_as_one_channel = False
obs_config.front_right_camera.masks_as_one_channel = False

obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL3
obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL3
obs_config.overhead_camera.render_mode = RenderMode.OPENGL3
obs_config.wrist_camera.render_mode = RenderMode.OPENGL3
obs_config.front_camera.render_mode = RenderMode.OPENGL3
obs_config.front_left_camera.render_mode = RenderMode.OPENGL3
obs_config.front_right_camera.render_mode = RenderMode.OPENGL3

demos = get_stored_demos(amount = 1,
                         dataset_root = DATASET_ROOT,
                         task_name='open_drawer',
                         variation_number=0,
                         image_paths=False,
                         obs_config=obs_config)

demo = demos[0]

# f_pcd_mat = demo[0].front_point_cloud
# # print(type(f_pcd))
# # print(f_pcd_mat.shape)
# f_pcd_mat= f_pcd_mat.reshape(512*512,3)
# # print(f_pcd_mat.shape)
# f_pcd = o3d.geometry.PointCloud()
# f_pcd.points = o3d.utility.Vector3dVector(f_pcd_mat)
# o3d.io.write_point_cloud("f_pcd.ply", f_pcd) 

# fl_pcd_mat = demo[0].front_left_point_cloud
# fl_pcd_mat= fl_pcd_mat.reshape(512*512,3)
# fl_pcd = o3d.geometry.PointCloud()
# fl_pcd.points = o3d.utility.Vector3dVector(fl_pcd_mat)
# o3d.io.write_point_cloud("fl_pcd.ply", f_pcd) 

# fr_pcd_mat = demo[0].front_right_point_cloud
# fr_pcd_mat= fr_pcd_mat.reshape(512*512,3)
# fr_pcd = o3d.geometry.PointCloud()
# fr_pcd.points = o3d.utility.Vector3dVector(fr_pcd_mat)
# o3d.io.write_point_cloud("fr_pcd.ply", fr_pcd) 

# ls_pcd_mat = demo[0].left_shoulder_point_cloud
# ls_pcd_mat= ls_pcd_mat.reshape(512*512,3)
# ls_pcd = o3d.geometry.PointCloud()
# ls_pcd.points = o3d.utility.Vector3dVector(ls_pcd_mat)
# o3d.io.write_point_cloud("ls_pcd.ply", ls_pcd) 

# rs_pcd_mat = demo[0].right_shoulder_point_cloud
# rs_pcd_mat= rs_pcd_mat.reshape(512*512,3)
# rs_pcd = o3d.geometry.PointCloud()
# rs_pcd.points = o3d.utility.Vector3dVector(rs_pcd_mat)
# o3d.io.write_point_cloud("rs_pcd.ply", rs_pcd) 

# oh_pcd_mat = demo[0].overhead_point_cloud
# oh_pcd_mat= oh_pcd_mat.reshape(512*512,3)
# oh_pcd = o3d.geometry.PointCloud()
# oh_pcd.points = o3d.utility.Vector3dVector(oh_pcd_mat)
# o3d.io.write_point_cloud("oh_pcd.ply", oh_pcd) 

# merged_pcd = f_pcd + fr_pcd + fl_pcd + ls_pcd + rs_pcd + oh_pcd
# merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.02)
# o3d.io.write_point_cloud("merged_pcd.ply", merged_pcd) 

# front_merged_pcd = f_pcd + fr_pcd + fl_pcd
# front_merged_pcd = front_merged_pcd.voxel_down_sample(voxel_size=0.01)
# o3d.io.write_point_cloud("front_merged_pcd_dense.ply", front_merged_pcd) 

ls_ext = demo[0].misc['left_shoulder_camera_extrinsics']
ls_int = demo[0].misc['left_shoulder_camera_intrinsics']

rs_ext = demo[0].misc['right_shoulder_camera_extrinsics']
rs_int = demo[0].misc['right_shoulder_camera_intrinsics']

oh_ext = demo[0].misc['overhead_camera_extrinsics']
oh_int = demo[0].misc['overhead_camera_intrinsics']

f_ext = demo[0].misc['front_camera_extrinsics']
f_int = demo[0].misc['front_camera_intrinsics']

fl_ext = demo[0].misc['front_left_camera_extrinsics']
fl_int = demo[0].misc['front_left_camera_intrinsics']

fr_ext = demo[0].misc['front_right_camera_extrinsics']
fr_int = demo[0].misc['front_right_camera_intrinsics']

print(ls_ext)
print(ls_int)

# x = np.linspace(-3, 3, 401)
# mesh_x, mesh_y = np.meshgrid(x, x)
# z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
# z_norm = (z - z.min()) / (z.max() - z.min())
# xyz = np.zeros((np.size(mesh_x), 3))
# xyz[:, 0] = np.reshape(mesh_x, -1)
# xyz[:, 1] = np.reshape(mesh_y, -1)
# xyz[:, 2] = np.reshape(z_norm, -1)
# print(xyz.shape)