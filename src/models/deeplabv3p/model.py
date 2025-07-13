from typing import Any, Literal, Optional, Iterable

import lightning as L
import numpy as np
import segmentation_models_pytorch as smp
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from pydantic import BaseModel, Field, ConfigDict
from torch.nn.modules.loss import _Loss
from torchvision.transforms import v2
from models import TorchFloatDTypes


class DeepLabV3PlusConfig(BaseModel):
    model_config = ConfigDict(validate_default=True, validate_assignment=True)
    input_size: tuple[int, int] = Field(
        default_factory=lambda _: (512, 512), exclude=True
    )
    strict_input_size: bool = Field(True, exclude=True)
    label2class: dict[int, str] = Field(exclude=True)

    # smp.DeepLabV3Plus parameters
    classes: int
    encoder_name: str = "resnet34"
    encoder_depth: Literal[3, 4, 5] = 5
    encoder_weights: Optional[str] = "imagenet"
    encoder_output_stride: Literal[8, 16] = 16
    decoder_channels: int = 256
    decoder_atrous_rates: Iterable[int] = Field(default_factory=lambda _: (12, 24, 36))
    decoder_aspp_separable: bool = True
    decoder_aspp_dropout: float = 0.5
    in_channels: int = 3
    activation: Optional[str] = None
    upsampling: int = 4
    aux_params: dict | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict, exclude=True)

"""
Computations (init).

Train Loop (training_step)

Validation Loop (validation_step)

Test Loop (test_step)

Prediction Loop (predict_step)

Optimizers and LR Schedulers (configure_optimizers)

"""


class DeepLabV3Plus(L.LightningModule):

    def __init__(self, config: DeepLabV3PlusConfig, loss_fn: _Loss = None):
        super().__init__()
        self.config = config
        self.model = smp.DeepLabV3Plus(
            **config.model_dump(),
            **config.kwargs,
        )
        params = smp.encoders.get_preprocessing_params(config.encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function for multi-class segmentation
        self.loss_fn = loss_fn or smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self._to_dtype = v2.ToDtype(self.dtype, scale=True)
        
    @property
    def input_size(self):
        return self.config.input_size

    @property
    def strict_input(self):
        return self.config.strict_input_size
    
    @strict_input.setter
    def strict_input(self,v:bool):
        assert isinstance(v,bool)
        self.config.strict_input_size = v
    
    @property
    def num_classes(self):
        return self.config.classes

    def sync_input_type(self, input_t: torch.Tensor, scale: bool = True) -> torch.Tensor:
        self._to_dtype.dtype = self.dtype
        self._to_dtype.scale = scale
        return self._to_dtype(input_t).to(device=self.device)

    def forward(self, images: torch.Tensor | np.ndarray) -> Any:
        """

        Args:
            images: Can be Tensor or numpy array, dim can be 3 or 4, shape can be (HxWxC) or (CxHxW).

        IMPORTANT: If the dtype is float, it must be already scaled to range 0-1, this method does not scale if it's
         originally float. Or if it's uint8, it must not be already scaled, this method will convert it to float and do scaling.
         Rule of thump: uint8 in range 0-255, or float in range 0-1.

        Returns:

        """
        shape = images.shape
        org_shape = shape
        assert len(shape) in [3, 4]

        # Only Uint8 will be converted to self.dtype AND SCALE
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        
        if len(shape) == 3:
            images = images.unsqueeze(0)
            shape = images.shape
        if shape[1] != 3:
            if shape[3] !=3:
                raise ValueError(f"Invalid shape {org_shape}")
            images = images.permute(0, 3, 1, 2)
            shape = images.shape
        if self.strict_input:
            if not shape[2:] == self.input_size:
                raise ValueError(f"Model is in strict input mode, so what image input height and width must be {self.input_size}. Got {shape[2:]}.")
        
        
        if images.dtype == torch.uint8:
            images = self.sync_input_type(images, scale=True)
        elif not images.dtype in TorchFloatDTypes:
            raise ValueError(f"Input dtype must be float or uint8, got {images.dtype}")
        else:
            images = images.to(device=self.device)

        images = (images - self.mean) / self.std
        return self.model(images)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int) -> STEP_OUTPUT:
        pred = self(batch[0])
        loss = self.loss_fn(pred, batch[1].to(torch.long) )
        self.log("train_loss", loss, on_step=True, logger=True )
        return loss
    
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.AdamW(self.parameters())