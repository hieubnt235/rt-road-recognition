from torch.utils.data import DataLoader
from datasets.camvid import CamVid
import lightning as L
import albumentations as A
from models.deeplabv3p import DeepLabV3Plus, DeepLabV3PlusConfig

config = DeepLabV3PlusConfig(
    input_size=(512, 512),
    classes=CamVid.num_classes,
    label2class= CamVid.label2class
)
model = DeepLabV3Plus(config)

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
    ]
)

train_dl = DataLoader(
    CamVid(augmentation=transform), batch_size=8, shuffle=True, num_workers=8
)

trainer = L.Trainer(max_epochs=5, log_every_n_steps=1)
trainer.fit(model=model, train_dataloaders=train_dl)
