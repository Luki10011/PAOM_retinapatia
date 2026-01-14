import sys 
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
from src.models.dr_model import DRLightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.datamodules.dr_module import RDDatamodule


if __name__ == "__main__":
    best_model = DRLightning.load_from_checkpoint(
        checkpoint_path=r"C:\PAOM\PAOM_retinapatia\checkpoints\best-epoch=20-val_acc=0.786.ckpt"
    )
    print(1)
