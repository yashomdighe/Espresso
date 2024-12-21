import torch
import torch.nn as nn

from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb
from lightning.pytorch.loggers import WandbLogger
from models.pointnet2_cls_ssg import PointNet2 as pointnet
from models.espressoV2 import EspressoV2

from data_scripts.espressov2_dataloader import EspressoDataset

from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":

    # model = EspressoV2.load_from_checkpoint("/home/ydighe/Developer/Espresso/weights/espresso_v1/espresso_v1-epoch=27-val_acc=0.00.ckpt", out_channels=3)
    model = EspressoV2(out_channels=3)

    print(f"Created model")

    train_set = EspressoDataset("train_paths.csv", )
    val_set = EspressoDataset("val_paths.csv", )

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    print("Obtained train set")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    print("Obtained val set")

    wandb_logger = WandbLogger()
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="Espresso",
        # name=f"v1",
        # Track hyperparameters and run metadata
        config={
            "max_epochs": 200,
        },
    )
    wandb.watch(model, log_freq=10, log="all")
    # saves top-K checkpoints based on "val_accuracy" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        dirpath=f"./weights/espresso_v2",
        filename="espresso_v2-{epoch:02d}-{val_loss:.2f}",
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
                        callbacks=callbacks,
                        fast_dev_run=False)

    # Train Model
    trainer.fit(model, train_loader, val_loader)