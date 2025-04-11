import torch
import torch.nn as nn

from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb
from lightning.pytorch.loggers import WandbLogger
from models.pointnet2_cls_ssg import PointNet2 as pointnet
from models.espressoV5_batched import EspressoV5

from data_scripts.espressov5_dataloader_batched import EspressoDataset, espresso_collate_fn

from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('medium')


def espresso_collate_fn_no_pad(batch):
    """
    Collate function that returns a batch as lists (no padding).
    Keeps variable-length tensors as lists, and stacks fixed-size data.
    """

    # Unzip each field
    (means3D, opacity, scales, rotations, shs, active_sh_degree,
     world_view_transform, full_proj_transform, camera_center,
     FovX, FovY, prompt, image, gt_means, pt_mask, im_mask) = zip(*batch)

    # Variable-size data → keep as list
    means3D = list(means3D)
    opacity = list(opacity)
    scales = list(scales)
    rotations = list(rotations)
    shs = list(shs)
    pt_mask = list(pt_mask)
    gt_means = list(gt_means)

    # Fixed-size data → stack
    active_sh_degree = torch.tensor(active_sh_degree)
    world_view_transform = torch.stack(world_view_transform)
    full_proj_transform = torch.stack(full_proj_transform)
    camera_center = torch.stack(camera_center)
    FovX = torch.tensor(FovX)
    FovY = torch.tensor(FovY)
    image = torch.stack(image)
    im_mask = torch.stack(im_mask)
    prompt = list(prompt)

    return (means3D, opacity, scales, rotations, shs, active_sh_degree,
            world_view_transform, full_proj_transform, camera_center,
            FovX, FovY, prompt, image, gt_means, pt_mask, im_mask)


if __name__ == "__main__":

    # model = EspressoV2.load_from_checkpoint("/home/ydighe/Developer/Espresso/weights/espresso_v1/espresso_v1-epoch=27-val_acc=0.00.ckpt", out_channels=3)
    version = "v5.4b"
    loss_type = "masked_mse + fast rigid (radius 1e-2)"
    model = EspressoV5(out_channels=3, version=version)

    print(f"Created model")

    train_set = EspressoDataset("train_paths2.csv", )
    val_set = EspressoDataset("val_paths2.csv", )

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=espresso_collate_fn)
    print("Obtained train set")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=espresso_collate_fn)
    print("Obtained val set")

    wandb_logger = WandbLogger()
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="EspressoV3",
        name=version,
        # Track hyperparameters and run metadata
        config={
            "max_epochs": 200,
            "loss_type": loss_type
        },
    )
    wandb.watch(model, log_freq=10, log="all")
    # saves top-K checkpoints based on "val_accuracy" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        dirpath=f"./weights/espresso_v4",
        filename="espresso_{version}-{epoch:02d}-{val_loss:.2f}",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.01, 
        patience=10, 
        verbose=True, 
        mode="min")
    
    # callbacks.append(early_stop_callback)
    callbacks = [checkpoint_callback, ModelSummary(max_depth=2), early_stop_callback]

    trainer = L.Trainer(accelerator="cuda", 
                        devices=1,
                        logger=wandb_logger,
                        max_epochs=200,
                        check_val_every_n_epoch=2,
                        log_every_n_steps=4,
                        callbacks=callbacks,
                        fast_dev_run=False)

    # Train Model
    trainer.fit(model, train_loader, val_loader)