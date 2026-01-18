import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from pytorch_lightning import Trainer

from datamodules.dr_module import RDDatamodule
from models.dr_model import DRLightning


def main():
    # -------------------------
    # Paths
    # -------------------------
    CHECKPOINT_PATH = Path("../../training_18012026/best-finetuned-epoch=03-val_acc=0.768.ckpt")

    OUTPUT_FILE = Path("./test_metrics.txt")

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    # -------------------------
    # Data
    # -------------------------
    datamodule = RDDatamodule()
    datamodule.setup()

    # -------------------------
    # Load model
    # -------------------------
    model = DRLightning.load_from_checkpoint(
        checkpoint_path=str(CHECKPOINT_PATH),
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )

    model.eval()

    # -------------------------
    # Trainer (test only)
    # -------------------------
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,      # disable logging
        enable_checkpointing=False,
    )

    # -------------------------
    # Test
    # -------------------------
    test_results = trainer.test(
        model=model,
        datamodule=datamodule,
        verbose=True,
    )

    # test_results is a list with one dict
    metrics = test_results[0]

    # -------------------------
    # Save metrics to file
    # -------------------------
    with open(OUTPUT_FILE, "w") as f:
        f.write("Test results – Diabetic Retinopathy Classification\n")
        f.write("=================================================\n\n")

        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")

    print(f"\n✅ Test metrics saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
