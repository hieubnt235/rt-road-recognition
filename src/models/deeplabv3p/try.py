from data.camvid import CamVidDataModule, CamVidDataModuleConfig, CamVidDatasetConfig
from models.deeplabv3p import DeepLabV3Plus
import lightning as L
model = DeepLabV3Plus.load_from_checkpoint(checkpoint_path="./models/DeepLabV3Plus/checkpoints/epoch=1-step=184-min_train_loss=0.49.ckpt")
camvid_dm = CamVidDataModule(
    CamVidDataModuleConfig(
        test=CamVidDatasetConfig( batch_size=4),
        predict=CamVidDatasetConfig( batch_size=4),
    )
)
outputs = L.Trainer().test(model,datamodule=camvid_dm)

print(outputs)