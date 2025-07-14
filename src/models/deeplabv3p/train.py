import albumentations as A
import lightning as L
from dotenv import find_dotenv, load_dotenv
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import (ModelCheckpoint, RichProgressBar, )
from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

from data.camvid import (CamVid, CamVidDataModule, CamVidDataModuleConfig, CamVidDatasetConfig, )
from models.deeplabv3p import DeepLabV3Plus, DeepLabV3PlusConfig

config = DeepLabV3PlusConfig(
    input_size=(512, 512), classes=CamVid.num_classes, label2class=CamVid.label2class
)

model = DeepLabV3Plus(config)
augmentation = A.Compose(
    [
        A.HorizontalFlip(p=0.5),  # todo
    ]
)
camvid_dm = CamVidDataModule(
    CamVidDataModuleConfig(
        train=CamVidDatasetConfig(augmentation=augmentation, batch_size=4),
        val=CamVidDatasetConfig(augmentation=augmentation, batch_size=4),
    )
)

env_file = find_dotenv(".env")
load_dotenv(env_file)


class TrainSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=env_file, extra="ignore")
    checkpoints_path: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info(self.model_dump_json(indent=4))


train_settings = TrainSettings()
dirpath = train_settings.checkpoints_path
if not dirpath.endswith("/"):
    dirpath += "/"
dirpath += f"{model.__class__.__name__}"
checkpoints_path = f"{dirpath}/checkpoints"

callbacks = [
    ModelCheckpoint(
        dirpath=checkpoints_path,
        filename="{epoch}-{step}-min_{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_on_train_epoch_end=True,
    ),
    ModelCheckpoint(
        dirpath=checkpoints_path,
        filename="{epoch}-{step}-min_{train_loss:.2f}",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        save_on_train_epoch_end=True,
    ),
    ModelCheckpoint(
        dirpath=checkpoints_path,
        filename="{epoch}-{step}-last",
        save_top_k=1,
        save_on_train_epoch_end=True,
    ),
    RichProgressBar(leave=True),
    # TQDMProgressBar(leave=True),
]

loggers = [
    pl_loggers.CSVLogger(dirpath, version=0),
    pl_loggers.TensorBoardLogger(dirpath, version=0),
]

trainer = L.Trainer(
    default_root_dir=dirpath, max_epochs=5, callbacks=callbacks, logger=loggers
)

trainer.fit(model=model, datamodule=camvid_dm)
