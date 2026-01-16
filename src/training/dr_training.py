import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from models.dr_model import DRLightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datamodules.dr_module import RDDatamodule

if __name__ == "__main__":
    datamodule = RDDatamodule()
    datamodule.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Urządzenie:", device)

    model = DRLightning(learning_rate = 1e-4) 
        
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",       # monitorujemy stratę walidacyjną
        dirpath="./checkpoints",  # folder na checkpointy
        filename="best-{epoch:02d}-{val_acc:.3f}",
        save_top_k=1,             # zapisujemy tylko najlepszy model
        mode="min"                # minimalizujemy val_loss
    )


    trainer = Trainer(
        max_epochs=200,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)