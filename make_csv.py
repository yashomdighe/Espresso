import numpy as np
import pandas as pd
import os 
from pathlib import Path
from random import shuffle

from models.gaussian_model import GaussianModel

if __name__ == "__main__":

    root = "/home/ydighe/Developer/datasets/gaussian-splatting/slide_block_to_target/variation_0/"
    paths = []
    count = 0
    # gt_epi_li =[]
    for epi in range(0, 99):
        episode_path = os.path.join(root, f"episode_{epi}")
        subdirs = sorted([int(f.path.split("/")[-1]) for f in os.scandir(episode_path) if f.is_dir() and f.path.split("/")[-1] != "splat"])
        gt_epi = subdirs[-1]
        # gt_epi_li.append(gt_epi)
        # print(gt_epi)
        final = gt_epi - (gt_epi % 10)
        for timestep in range(0, min(final, 90), 10):

            # continue
            # print(os.listdir(episode_path))
            timestep_path = os.path.join(episode_path, str(timestep))
            splat_path = os.path.join(timestep_path, "splat/point_cloud/iteration_7000/point_cloud.ply")
            splat = GaussianModel(3)
            splat.load_ply(splat_path)

            # print(splat.get_xyz.size())
            # if splat.get_xyz.size()[0] <= 131072:
                # count += 1
            paths.append(
                {
                    "input_splat": splat_path,
                    "gt_im": os.path.join(episode_path, str(gt_epi), "images"),
                    "gt_pcd": os.path.join(episode_path, str(gt_epi), "splat/point_cloud/iteration_7000/point_cloud.ply"),

                }
            )

            # break
        # break
            # subdirs = sorted([int(f.path.split("/")[-1]) for f in os.scandir(episode_path) if f.is_dir() and f.path.split("/")[-1] != "splat"])
            # subdirs.pop('splat')
            # for dir in subdirs:
            #     if dir % 10 == 0:
            #         paths.append(
            #             {
            #                 "input_splat": os.path.join(root, f"episode_{epi}", f"{dir}", "splat/point_cloud/iteration_7000"),
            #                 "gt": os.path.join(root, f"episode_{epi}", f"{dir}", "images"),

            #             }
            #         )
    # print(count)
    # print(min(gt_epi_li))    
    print(f"Total Demos :{len(paths)}")
    shuffle(paths)

    train_paths = paths[0:int(0.7*len(paths))]
    test_paths = paths[int(0.7*len(paths))+1:int(0.9*len(paths))]
    val_paths = paths[int(0.9*len(paths))+1:]
    
    print(f"Train Demo: {len(train_paths)}")
    print(f"Test Demo: {len(test_paths)}")
    print(f"Val Demo: {len(val_paths)}")
    train_df = pd.DataFrame(train_paths)
    train_df.to_csv("train_paths.csv", index= False)
    test_df = pd.DataFrame(test_paths)
    test_df.to_csv("test_paths.csv", index= False)
    val_df = pd.DataFrame(val_paths)
    val_df.to_csv("val_paths.csv", index= False)
        
