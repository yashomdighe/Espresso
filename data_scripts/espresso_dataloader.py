# TO DO modify the network and dataloader to not use a Gaussian model object but just use the individual tensors
# means, sh_degree etc

import os 
import torch 
import pandas as pd
import numpy as np
from numpy.random import default_rng
import collections
import math
from PIL import Image
import cv2
from torch.utils.data import Dataset

from models.gaussian_model import GaussianModel

from torchvision.transforms import ToTensor

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])



def pad_tensor(tensor, target_size, pad_value, mask=False):
    current_size = tensor.size(0)

    if current_size < target_size:
        padding_shape = (target_size - current_size, *tensor.shape[1:])
        padding = torch.full(padding_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=0)
    
    if mask:
        mask = torch.cat([torch.ones(current_size, dtype=torch.long, device=tensor.device),
                        torch.zeros(target_size - current_size, dtype=torch.long, device=tensor.device)])
        
        return tensor, mask
        
    return tensor

def pad_rotations(rotations, target_size):
    """
    Pads the `rotations` tensor along the first dimension with [1, 0, 0, 0] until it reaches `target_size`.
    
    Args:
        rotations (torch.Tensor): Input tensor of shape [N, 4].
        target_size (int): The desired size of the first dimension after padding.
    
    Returns:
        torch.Tensor: Padded tensor of shape [target_size, 4].
    """
    current_size = rotations.size(0)
    if current_size < target_size:
        # Create padding tensor with [1, 0, 0, 0]
        rotation_padding = torch.tensor([1, 0, 0, 0], dtype=rotations.dtype, device=rotations.device)
        padding = rotation_padding.repeat(target_size - current_size, 1)
        padded_rotations = torch.cat([rotations, padding], dim=0)
    else:
        padded_rotations = rotations

    return padded_rotations


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

class ImageCls(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

class EspressoDataset(Dataset):
    """RL Bench data loader for project codenamed espresso"""

    def __init__(self, csv_file, target_size):

        self.rng = default_rng(seed=1234)
        self.intrinsics = ["PINHOLE", 128, 128, -175.83856040078922, -175.83856040078922, 64.0, 64.0]
        self.paths_frame = pd.read_csv(csv_file)
        self.cam_ids = np.array([1,2,3,4,5,6])
        self.zfar = 100.0
        self.znear = 0.01
        self.cameras = {
            1: ("f", (1, 0.9990482245960659, -0.04361931832712903, 5.878443255279756e-08, 3.491770844128399e-07, 1.1109524872083874e-06, -1.2570885722692615, 1.4651376416239166, 1)),
            2: ("fl",(2, 0.9699897928399164, -0.16937036190067292, 0.17298730302352386, 0.022558264309530347, 0.11288327146035265, -1.1113956865916623, 2.250656826261323, 2)),
            3: ("fr",(3, 0.9623519203233675, -0.14782955358075406, -0.2113777398514418, 0.08570096635008294, 0.05164049378742085, -1.118523786802849, 2.249362977734404, 3)),
            4: ("ls",(4, -0.165335774570635, -0.09977885323176063, 0.8355496472706789, -0.5143588718402037, 0.48242383874789274, -0.7544217533496278, 2.029270850816078, 4)),
            5: ("rs",(5, 0.16533572606187297, 0.09977866008346868, 0.835549718224285, -0.5143588096405648, -0.4824232333668008, -0.7544216654080442, 2.0292702804320424, 5)),
            6: ("oh",(6, 8.524354550857348e-09, -1.1953473994618564e-07, 0.8433915474061249, -0.5372994488773349, 2.669880156539348e-07, -0.7958520982871833, 2.3100895563512878, 6))
        }
        self.target_size = target_size
        self.tensor_transform = ToTensor()

    def get_extrinsics(self, cam_id):
        image_name, params = self.cameras[cam_id]
        image_id = int(params[0])
        qvec = np.array(tuple(map(float, params[1:5])))
        tvec = np.array(tuple(map(float, params[5:8])))
        camera_id = cam_id
        xys = np.column_stack([tuple(map(float, params[0::3])),
                                       tuple(map(float, params[1::3]))])
        point3D_ids = np.array(tuple(map(int, params[2::3])))

        return ImageCls(id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=f"{image_name}.png",
                    xys=xys, point3D_ids=point3D_ids)

    def get_intrinsics(self, cam_id):

        camera_id = cam_id
        model = self.intrinsics[0]
        width = int(self.intrinsics[2])
        height = int(self.intrinsics[3])
        params = np.array(tuple(map(float, self.intrinsics[4:])))
        
        return Camera(id=camera_id, model=model,
                      width=width, height=height,
                      params=params)

    def get_rasterization_settings(self, extr, intr):
        height = intr.height
        width = intr.width

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        focal_length_x = intr.params[0]
        focal_length_y = intr.params[1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 0.1)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]  
        
        # print(f"fovx :{FovX}")
        # print(f"fovy :{FovY}")
        return world_view_transform, full_proj_transform, camera_center, FovX, FovY

    def __len__(self):
        return len(self.paths_frame)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()


        splat = GaussianModel(3)
        splat_path = os.path.join(self.paths_frame.iloc[idx, 0], "point_cloud.ply")
        splat.load_ply(splat_path)

        means3D, mask = pad_tensor(splat.get_xyz, self.target_size, 0,  True)
        opacity = pad_tensor(splat.get_opacity, self.target_size, 0)
        scales = pad_tensor(splat.get_scaling, self.target_size, 0)
        shs = pad_tensor(splat.get_features, self.target_size, 0)
        rotations = pad_rotations(splat.get_rotation, self.target_size)
        active_sh_degree = splat.active_sh_degree

        # print(f"Means: {means3D.size(), means3D.dtype}")
        # print(f"Opacity: {opacity.size(), opacity.dtype}")
        # print(f"Scales: {scales.size(), scales.dtype}")
        # print(f"rotations: {rotations.size(), rotations.dtype}")
        # print(f"shs: {shs.size(), shs.dtype}")
        # print(f"active_sh_degree: {active_sh_degree}")
        # print("##################################")
        # # print(f"Means: {means3D.size()}")
        # print(f"Opacity: {opacity}")
        # print(f"Scales: {scales}")
        # print(f"rotations: {rotations}")
        # print(f"shs: {shs}")



        cam = self.rng.choice(self.cam_ids)
        img_path = os.path.join(self.paths_frame.iloc[idx, 1], f"{self.cameras[cam][0]}.png")
        # img = Image.open(img_path)
        # img
        # print(img)
        image =  self.tensor_transform(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        # cv2.imwrite("test1.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # image = self.tensor_transform(img)
        ext = self.get_extrinsics(cam)
        intr = self.get_intrinsics(cam)
        
        world_view_transform, full_proj_transform, camera_center, FovX, FovY = self.get_rasterization_settings(ext, intr)
        # print(image.size())
        return means3D, mask, opacity, scales, rotations, shs, active_sh_degree, world_view_transform, full_proj_transform, camera_center, FovX, FovY, "slide red block to green target", image