import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from datamodules.dr_module import RDDatamodule
from models.dr_model import DRLightning
from models.en_pretrained import build_efficientnet_b4, unfreeze_backbone


if __name__ == "__main__":
    # -------------------------
    # Data
    # -------------------------
    datamodule = RDDatamodule()
    datamodule.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Urządzenie:", device)

    # -------------------------
    # Stage 1: frozen backbone
    # -------------------------
    backbone = build_efficientnet_b4(
        num_classes=5,
        freeze_backbone=True,
    )

    model = DRLightning(
        model=backbone,
        num_classes=5,
        learning_rate=1e-3,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename="best-frozen-{epoch:02d}-{val_acc:.3f}",
        save_top_k=1,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)

    # -------------------------
    # Stage 2: fine-tuning
    # -------------------------
    print("\nUnfreezing backbone for fine-tuning...\n")
    unfreeze_backbone(model.model)

    checkpoint_callback_ft = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename="best-finetuned-{epoch:02d}-{val_acc:.3f}",
        save_top_k=1,
        mode="min",
    )

    trainer_ft = Trainer(
        max_epochs=200,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback_ft],
        log_every_n_steps=10,
    )

    trainer_ft.fit(model, datamodule=datamodule)
    trainer_ft.test(model, datamodule=datamodule)
