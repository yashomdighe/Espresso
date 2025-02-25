import torch
import torch.nn as nn

from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import wandb
from lightning.pytorch.loggers import WandbLogger
from models.pointnet2_cls_ssg import PointNet2 as pointnet
from models.espresso import Espresso

from data_scripts.espresso_dataloader import EspressoDataset

from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    
    d_model = 512
    # task_desc = "slide red block to green target"
    pretrained_model = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval() 
    text_encoder.to("cuda")
    print(f"Created text encoder")

    pcd_encoder = pointnet(num_class=40, normal_channel=False)
    checkpoint = torch.load("cls_ssg.pth", weights_only=False)
    pcd_encoder.load_state_dict(checkpoint['model_state_dict'])
    pcd_encoder.eval()
    pcd_encoder.requires_grad_(False)
    pcd_encoder.to("cuda")
    pcd_encoder.fc3 = nn.Linear(256, d_model)
    print(f"Created pcd encoder")

    model = Espresso(tokenizer, text_encoder, pcd_encoder, d_model)

    print(f"Created model")

    train_set = EspressoDataset("train_paths.csv", target_size=131072)
    val_set = EspressoDataset("val_paths.csv", target_size=131072)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    print("Obtained train set")

    # Trainer
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

    # saves top-K checkpoints based on "val_accuracy" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="val_loss",
        mode="min",
        dirpath=f"./weights/espresso_v1",
        filename="espresso_v1-{epoch:02d}-{val_acc:.2f}",
    )


    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.01, patience=10, verbose=True, mode="min")
    
    # callbacks.append(early_stop_callback)
    callbacks = [checkpoint_callback, ModelSummary(max_depth=2), early_stop_callback]

    trainer = L.Trainer(accelerator="auto", 
                        devices=1,
                        logger=wandb_logger,
                        max_epochs=200,
                        check_val_every_n_epoch=2,
                        callbacks=callbacks,
                        fast_dev_run=False)

    # Train Model
    trainer.fit(model, train_loader, val_loader)