__all__ = [
    "CamVidDataModule",
    "CamVidDataModuleConfig",
    "CamVidDatasetConfig",
    "CamVid",
]
import zipfile
from math import ceil
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from lightning.pytorch.trainer.states import TrainerFn
from loguru import logger
import albumentations as A
import cv2
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from minio import Minio
from minio.helpers import ProgressType
from pydantic import BaseModel, PositiveInt, ConfigDict, Field
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataset import _T_co
import tqdm
from .meta import classes, split_dirs, class2rgb, camvid_name
from ..settings import DatasetSettings

ds_settings = DatasetSettings()


camvid_path = Path(ds_settings.dataset_path).joinpath(camvid_name)

rgb2class = {v: k for k, v in class2rgb.items()}
class2label = {c: i for i, c in enumerate(classes)}
label2class = {i: c for i, c in enumerate(classes)}
num_classes = len(classes)


class CamVid(Dataset):
    """
    classes: class name, a string
    label: unique integer represent the class.
    mask: rgb image, all colors are the rgb represent of classes
    target: image of labels
    """

    rgb2class = rgb2class
    class2label = class2label
    label2class = label2class
    num_classes = num_classes

    def __init__(
        self,
        root: Path = camvid_path,
        split: Literal["train", "test", "val"] = "train",
        load_size: tuple[int, int] | None = (512, 512),
        augmentation: A.Compose | None = None,
    ):
        """

        Args:
            root:
            split:
            load_size: Image will be resized by this value after loaded intermediately.
            augmentation: augmentation image after loaded and resized.
        """
        self.root = root
        self.dirs = split_dirs[split]
        self.data_dir = self.root.joinpath(self.dirs[0])
        self.labels_dir = self.root.joinpath(self.dirs[1])

        self.data_files: list[str] = []
        self.label_files: list[str] = []

        for file in self.data_dir.iterdir():
            if file.is_file():
                self.data_files.append(file.as_posix())
                fn, ext = file.name.split(".")
                self.label_files.append(
                    self.labels_dir.joinpath(f"{fn}_L.{ext}").as_posix()
                )  # see images folder for details

        self.augmentation = augmentation
        self.load_size = load_size

    @classmethod
    def mask2target(cls, mask: np.ndarray) -> np.ndarray:
        assert mask.shape[2] == 3

        target = np.zeros(mask.shape[:2], dtype=np.uint8)
        for rgb, c in cls.rgb2class.items():
            matches = np.all(mask == rgb, axis=-1)
            target[matches] = cls.class2label[c]
        return target

    @classmethod
    def target2mask(cls, target: np.ndarray) -> np.ndarray:
        target = target.squeeze()
        mask = np.zeros(target.shape + (3,), dtype=np.uint8)
        for rgb, c in cls.rgb2class.items():
            label = cls.class2label[c]
            matches = target == label
            mask[matches] = rgb
        return mask

    def load_image(self, file: str):
        image = cv2.imread(file)
        if self.load_size:
            image = cv2.resize(image, self.load_size)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int) -> _T_co:
        image = self.load_image(self.data_files[index])
        mask = self.load_image(self.label_files[index])
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image = sample["image"]
            mask = sample["mask"]

        return image, self.mask2target(mask).astype(np.long)

    def __len__(self):
        return len(self.data_files)

    def show_pairs(
        self,
        n=25,
        n_per_row: int = 2,
        figsize=(20, 10),
        *,
        images_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        """

        Args:
            n:
            n_per_row:
            figsize:
            images_pairs: a pair of (image, target) # Note: receive target, not mask.

        Returns:

        """
        images_pair = images_pairs or []
        num_images = min(n, len(self))
        total_images = num_images + len(images_pair)
        nrows = ceil(total_images / n_per_row)
        fig, axes = plt.subplots(nrows, n_per_row, figsize=figsize)
        axes = axes.flatten()

        def img_pair():
            for r, t in images_pair:
                assert isinstance(r, np.ndarray) and isinstance(t, np.ndarray)
                m = self.target2mask(t)
                assert r.shape == m.shape
                yield r, m
            for i in range(num_images):
                r, target = self[i]
                m = self.target2mask(target)
                yield r, m

        for i, (ri, mi) in enumerate(img_pair()):
            raw_np = np.array(ri) if not isinstance(ri, np.ndarray) else ri
            mask_np = np.array(mi) if not isinstance(mi, np.ndarray) else mi
            concatenated_image = np.hstack((raw_np, mask_np))

            ax = axes[i]
            ax.imshow(concatenated_image)
            ax.set_title(f"Size {raw_np.shape}")
            ax.axis("off")  # Hide axes ticks and labels

        # Turn of the redundant
        for j in range(total_images, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(
            "Dataset Visualization: Raw Image (Left) & Mask (Right)", fontsize=16
        )

        plt.tight_layout()
        plt.show()


class CamVidDatasetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    batch_size: PositiveInt = 16
    load_size: tuple[int, int] | None = (512, 512)
    augmentation: A.Compose | None = Field(None, exclude=True)
    num_workers: PositiveInt = 8


class CamVidDataModuleConfig(BaseModel):
    train: CamVidDatasetConfig = CamVidDatasetConfig()
    test: CamVidDatasetConfig = CamVidDatasetConfig()
    val: CamVidDatasetConfig = CamVidDatasetConfig()
    predict: CamVidDatasetConfig = CamVidDatasetConfig()
    camvid_path: Path = camvid_path


class _MinioDownloadProgress(ProgressType):
    def __init__(self, desc=None):
        self._tqdm = tqdm.tqdm(desc=desc)
        self.object_name = None

    def set_meta(self, object_name: str, total_length: int):
        self.object_name = object_name
        self._tqdm.total = total_length

    def update(self, length: int):
        self._tqdm.update(length)


class CamVidDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: CamVidDataModuleConfig | None = None,
    ):
        self.config = config or CamVidDataModuleConfig()
        self.train_ds: CamVid | None = None
        self.test_ds: CamVid | None = None
        self.val_ds: CamVid | None = None
        self.predict_ds: CamVid | None = None
        super().__init__()

    @property
    def camvid_path(self):
        return self.config.camvid_path

    def prepare_data(self) -> None:
        if not self.is_valid_camvid_path():
            logger.info(f"'{self.camvid_path}' does not exists.")
            zip_file = f"{camvid_name}.zip"
            if not (
                local_ds_zip := Path(ds_settings.dataset_path).joinpath(zip_file)
            ).exists():
                logger.info(
                    f"'{local_ds_zip}' does not exists. Try to download from '{ds_settings.dataset_zip_path_s3}'"
                )
                try:
                    parsed_url = urlparse(ds_settings.aws_endpoint_url_s3)
                    client = Minio(
                        parsed_url.netloc,
                        access_key=ds_settings.aws_access_key_id,
                        secret_key=ds_settings.aws_secret_access_key,
                        secure=(parsed_url.scheme == "https"),
                    )
                except Exception as e:
                    raise ValueError(f"Cannot init S3 client. {e}")

                bn = ds_settings.bucket_name
                s3_ds_zip = (
                    Path(ds_settings.dataset_path_s3).joinpath(zip_file).as_posix()
                )
                try:
                    assert client.bucket_exists(bn)
                    assert client.stat_object(bn, s3_ds_zip)
                except Exception as e:
                    raise ValueError(f"'{s3_ds_zip}' or {bn} does not exists")

                client.fget_object(
                    ds_settings.bucket_name,
                    s3_ds_zip,
                    local_ds_zip.as_posix(),
                    progress=_MinioDownloadProgress(
                        f"Downloading '{zip_file}' from '{ds_settings.dataset_zip_path_s3}'"
                    ),
                )

            assert local_ds_zip.exists()
            logger.info(f"Detected '{local_ds_zip}'. Try to extract.")
            with zipfile.ZipFile(local_ds_zip.as_posix(), "r") as zf:
                for member in tqdm.tqdm(
                    zf.infolist(),
                    desc=f"Extracting {zip_file} to {ds_settings.dataset_path}",
                ):
                    zf.extract(member, ds_settings.dataset_path)
        assert self.is_valid_camvid_path()
        logger.info(f"'{self.camvid_path}' exists.")

    def is_valid_camvid_path(self):
        return self.camvid_path.exists() and self.camvid_path.is_dir()

    def setup(self, stage: TrainerFn):
        """
        Args:
            stage:
            - If stage is val, load val dataset
            - If stage is fit, load both val and fit datasets
            - If stage is test or predict, load both test or predict datasets
        
        Raises:
            ValueError: If stage does not match anything.
        """
        if not isinstance(stage, TrainerFn):
            raise ValueError(f"Invalid stage: '{stage}'")
        
        if stage in [TrainerFn.VALIDATING, TrainerFn.FITTING]:
            self.val_ds = CamVid(
                self.camvid_path,
                split="test",
                load_size=self.config.val.load_size,
                augmentation=self.config.val.augmentation,
            )

            if stage == TrainerFn.FITTING:
                self.train_ds = CamVid(
                    self.camvid_path,
                    split="train",
                    load_size=self.config.train.load_size,
                    augmentation=self.config.train.augmentation,
                )

        elif stage in [TrainerFn.TESTING, TrainerFn.PREDICTING]:
            test_ds = CamVid(
                self.camvid_path,
                split="test",
                load_size=self.config.test.load_size,
                augmentation=self.config.test.augmentation,
            )
            self.test_ds, self.predict_ds = random_split(
                test_ds, [0.9, 0.1], torch.Generator().manual_seed(42)
            )
        logger.debug(f"'{self.__class__.__name__}' setup for stage '{stage}'.")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            **self.config.train.model_dump(include={"batch_size", "num_workers"}),
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            **self.config.val.model_dump(include={"batch_size", "num_workers"}),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            **self.config.test.model_dump(include={"batch_size", "num_workers"}),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds,
            **self.config.predict.model_dump(include={"batch_size", "num_workers"}),
        )
