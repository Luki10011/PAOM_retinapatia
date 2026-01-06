import os
from pathlib import Path
import zipfile
import multiprocessing as mp
import shutil
import pandas as pd
from tqdm import tqdm


class DRPreprocessor:
    KAGGLE_COMPETITION = "aptos2019-blindness-detection"
    NUM_CLASSES = 5

    def __init__(
        self,
        output_root: str = "./data",
        should_preprocess: bool = False,
        num_workers: int = 4,
    ):
        self.output_root = Path(output_root)
        self.rawdata_root = self.output_root / "raw"
        self.should_preprocess = should_preprocess
        self.kaggle_path = ".kaggle"
        if self.should_preprocess:
            self.processed_data_root = self.output_root / "processed"
        self.num_workers = num_workers

    def _ensure_directories(self):
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.rawdata_root.mkdir(parents=True, exist_ok=True)

        if self.should_preprocess:
            self.processed_data_root.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Output root: {self.output_root.resolve()}")
        print(f"[INFO] Raw data directory: {self.rawdata_root.resolve()}")

    def _dataset_already_present(self) -> bool:
        return (
            (self.rawdata_root / "train.csv").exists()
            and (self.rawdata_root / "train_images").exists()
        )

    def _check_kaggle_credentials(self) -> bool:
        kaggle_json = Path(".kaggle/kaggle.json")

        if not kaggle_json.exists():
            print("[ERROR] kaggle.json not found.")
            return False

        os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_json.parent)
        return True

    def _download_dataset(self):
        if self._dataset_already_present():
            print("[INFO] Dataset already present. Skipping download.")
            return

        if not self._check_kaggle_credentials():
            raise RuntimeError("Missing Kaggle credentials.")

        print("[INFO] Downloading dataset via Kaggle CLI...")
        cmd = f"kaggle competitions download -c {self.KAGGLE_COMPETITION} -p {self.rawdata_root.resolve()}"
        if os.system(cmd) != 0:
            raise RuntimeError("Kaggle download failed.")

        for zip_path in self.rawdata_root.glob("*.zip"):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.rawdata_root)
            zip_path.unlink()

        print("[INFO] Dataset downloaded and extracted.")

    def _processed_data_exists(self) -> bool:
        """
        Checks whether class folders 0-4 already exist and contain images.
        """
        if not self.processed_data_root.exists():
            return False

        for cls in range(self.NUM_CLASSES):
            class_dir = self.processed_data_root / str(cls)
            if not class_dir.exists():
                return False
            if not any(class_dir.glob("*.png")):
                return False

        return True

    def _process_single_row(self, row):
        id_code = row["id_code"]
        diagnosis = str(row["diagnosis"])

        src = self.rawdata_root / "train_images" / f"{id_code}.png"
        dst_dir = self.processed_data_root / diagnosis
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{id_code}.png"

        if src.exists():
            shutil.copy2(src, dst)

    def _preprocess_dataset(self):
        if self._processed_data_exists():
            print("[INFO] Processed dataset already exists. Skipping preprocessing.")
            return

        csv_path = self.rawdata_root / "train.csv"
        df = pd.read_csv(csv_path)

        print(f"[INFO] Preprocessing {len(df)} images...")

        rows = [row for _, row in df.iterrows()]

        with mp.Pool(self.num_workers) as pool:
            list(
                tqdm(
                    pool.imap_unordered(self._process_single_row, rows),
                    total=len(rows),
                    desc="Preprocessing images",
                )
            )

        print("[INFO] Preprocessing completed successfully.")

    def run(self):
        self._ensure_directories()
        self._download_dataset()

        if self.should_preprocess:
            self._preprocess_dataset()
        else:
            print("[INFO] Preprocessing disabled.")



if __name__ == "__main__":
    preprocessor = DRPreprocessor(
        output_root="./data",
        should_preprocess=True,
        num_workers=mp.cpu_count(),
    )
    preprocessor.run()
