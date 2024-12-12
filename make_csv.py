import numpy as np
import pandas as pd
import os 
from pathlib import Path
from random import shuffle
if __name__ == "__main__":

    root = "/home/ydighe/Developer/datasets/gaussian-splatting/slide_block_to_target/variation_0/"
    paths = []
    for epi in range(10, 21):
        episode_path = os.path.join(root, f"episode_{epi}")
        # print(os.listdir(episode_path))
        subdirs = sorted([int(f.path.split("/")[-1]) for f in os.scandir(episode_path) if f.is_dir() and f.path.split("/")[-1] != "splat"])
        # subdirs.pop('splat')
        for dir in subdirs:
            if dir % 10 == 0:
                paths.append(
                    {
                        "input_splat": os.path.join(root, f"episode_{epi}", f"{dir}", "splat/point_cloud/iteration_7000"),
                        "gt": os.path.join(root, f"episode_{epi}", f"{dir}", "images"),

                    }
                )
    
    print(len(paths))
    shuffle(paths)

    train_paths = paths[0:int(0.7*len(paths))]
    test_paths = paths[int(0.7*len(paths))+1:int(0.9*len(paths))]
    val_paths = paths[int(0.9*len(paths))+1:]
    
    print(len(train_paths))
    print(len(test_paths))
    print(len(val_paths))
    train_df = pd.DataFrame(paths)
    train_df.to_csv("train_paths.csv", index= False)
    test_df = pd.DataFrame(paths)
    test_df.to_csv("test_paths.csv", index= False)
    val_df = pd.DataFrame(paths)
    val_df.to_csv("val_paths.csv", index= False)
        
