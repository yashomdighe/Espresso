import torch
import torch.nn as nn

from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb
from lightning.pytorch.loggers import WandbLogger
from models.pointnet2_cls_ssg import PointNet2 as pointnet
from models.espressoV5_multiview import EspressoV5

from data_scripts.espressov5_dataloader_multiview import EspressoDataset

from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":

    # model = EspressoV2.load_from_checkpoint("/home/ydighe/Developer/Espresso/weights/espresso_v1/espresso_v1-epoch=27-val_acc=0.00.ckpt", out_channels=3)
    version = "v5_7_multiview"
    loss_type = "im_loss + masked_mse + new rigid (radius 1e-2)"
    model = EspressoV5(out_channels=3, version=version)

    print(f"Created model")

    train_set = EspressoDataset("train_paths2.csv", )
    val_set = EspressoDataset("val_paths2.csv", )

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8)
    print("Obtained train set")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=8)
    print("Obtained val set")

    wandb.finish()
    wandb.login()
    wandb_logger = WandbLogger(
        project="EspressoV3",
        name=version,
        # Track hyperparameters and run metadata
        config={
            "max_epochs": 200,
            "loss_type": loss_type
        },
        log_model="all"
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        dirpath=f"./weights/espresso_{version}",
        filename="espresso_{version}-{epoch:02d}-{val_loss:.2f}",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.01, 
        patience=10, 
        verbose=True, 
        mode="min")
    
    # callbacks.append(early_stop_callback)
    callbacks = [checkpoint_callback, 
                 early_stop_callback,
                 ]

    trainer = L.Trainer(accelerator="gpu", 
                        devices=[1],
                        logger=wandb_logger,
                        max_epochs=200,
                        check_val_every_n_epoch=2,
                        log_every_n_steps=4,
                        callbacks=callbacks,
                        enable_model_summary="False",
                        fast_dev_run=False)

    # Train Model
    trainer.fit(model, train_loader, val_loader)