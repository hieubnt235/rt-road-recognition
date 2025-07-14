from typing import Any, Literal, Sequence

import lightning as L
import numpy as np
import segmentation_models_pytorch as smp
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from pydantic import BaseModel, Field, ConfigDict
from torch.nn.modules.loss import _Loss
from torchvision.transforms import v2
from typing_extensions import TypedDict


class DeepLabV3PlusConfig(BaseModel):
    model_config = ConfigDict(validate_default=True)
    input_size: tuple[int, int] = Field(
        default_factory=lambda _: (512, 512), exclude=True
    )
    strict_input_size: bool = Field(True, exclude=True)
    label2class: dict[int, str] = Field(exclude=True)

    # smp.DeepLabV3Plus parameters
    classes: int
    encoder_name: str = "resnet34"
    encoder_depth: Literal[3, 4, 5] = 5
    encoder_weights: str | None = "imagenet"
    encoder_output_stride: Literal[8, 16] = 16
    decoder_channels: int = 256
    decoder_atrous_rates: Sequence[int] = Field(default_factory=lambda _: (12, 24, 36))
    decoder_aspp_separable: bool = True
    decoder_aspp_dropout: float = 0.5
    in_channels: int = 3
    activation: str | None = None
    upsampling: int = 4
    aux_params: dict | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict, exclude=True)


class StepMetrics(TypedDict):
    loss: torch.Tensor
    tp: torch.LongTensor
    fp: torch.LongTensor
    fn: torch.LongTensor
    tn: torch.LongTensor


class DeepLabV3Plus(L.LightningModule):

    def __init__(self, config: DeepLabV3PlusConfig, loss_fn: _Loss = None):
        super().__init__()
        self.save_hyperparameters("config")
        self.model = smp.DeepLabV3Plus(
            **self.config.model_dump(),
            **self.config.kwargs,
        )

        # Loss function for multi-class segmentation
        self.loss_fn = loss_fn or smp.losses.DiceLoss(
            smp.losses.MULTICLASS_MODE, from_logits=True
        )

        params = smp.encoders.get_preprocessing_params(config.encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Step metrics tracking
        self.train_step_outputs: list[StepMetrics] = []
        self.val_step_outputs: list[StepMetrics] = []
        self.test_step_outputs: list[StepMetrics] = []

        self._to_dtype = v2.ToDtype(self.dtype, scale=True)

    @property
    def config(self) -> DeepLabV3PlusConfig:
        return self.hparams.config

    @property
    def input_size(self):
        return self.config.input_size

    @property
    def strict_input(self):
        return self.config.strict_input_size

    @strict_input.setter
    def strict_input(self, v: bool):
        assert isinstance(v, bool)
        self.config.strict_input_size = v

    @property
    def num_classes(self):
        return self.config.classes

    def sync_input_type(
        self, input_t: torch.Tensor, scale: bool = True
    ) -> torch.Tensor:
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
            if shape[3] != 3:
                raise ValueError(f"Invalid shape {org_shape}")
            images = images.permute(0, 3, 1, 2)
            shape = images.shape
        if self.strict_input:
            if not shape[2:] == self.input_size:
                raise ValueError(
                    f"Model is in strict input mode, so what image input height and width must be {self.input_size}. Got {shape[2:]}."
                )

        if images.dtype == torch.uint8:
            images = self.sync_input_type(images, scale=True)
        elif not torch.is_floating_point(images):
            raise ValueError(f"Input dtype must be float or uint8, got {images.dtype}")
        else:
            images = images.to(device=self.device)
        images = (images - self.mean) / self.std
        return self.model(images)

    # noinspection PyTypeChecker
    def make_step_metrics(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> StepMetrics:
        targets = targets.long()
        pred: torch.Tensor = self(images)
        loss = self.loss_fn(pred, targets)

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred.argmax(dim=1), targets, mode="multiclass", num_classes=self.num_classes
        )

        return StepMetrics(loss=loss, tp=tp, fp=fp, fn=fn, tn=tn)

    # noinspection PyTypeChecker,PyMethodMayBeStatic
    def make_epoch_metrics(
        self, prefix: Literal["train", "test", "val"], metric_list: list[StepMetrics]
    ) -> dict:
        tp = torch.cat([m["tp"] for m in metric_list], 0)
        fp = torch.cat([m["fp"] for m in metric_list], 0)
        fn = torch.cat([m["fn"] for m in metric_list], 0)
        tn = torch.cat([m["tn"] for m in metric_list], 0)

        return {
            f"{prefix}_loss": torch.stack([m["loss"] for m in metric_list]).mean(),
            f"{prefix}_tp": tp.to(torch.float).mean(),
            f"{prefix}_fp": fp.to(torch.float).mean(),
            f"{prefix}_fn": fn.to(torch.float).mean(),
            f"{prefix}_tn": tn.to(torch.float).mean(),
            f"{prefix}_dataset_iou": smp.metrics.iou_score(
                tp, fp, fn, tn, reduction="micro"
            ),
            f"{prefix}_per_image_iou": smp.metrics.iou_score(
                tp, fp, fn, tn, reduction="micro-imagewise"
            ),
            f"{prefix}_dataset_f1": smp.metrics.f1_score(
                tp, fp, fn, tn, reduction="micro"
            ),
            f"{prefix}_per_image_f1": smp.metrics.f1_score(
                tp, fp, fn, tn, reduction="micro-imagewise"
            ),
        }

    def get_step_metrics_list(
        self, prefix: Literal["train", "test", "val"]
    ) -> list[StepMetrics]:
        l: list[StepMetrics] = getattr(self, f"{prefix}_step_outputs")
        assert isinstance(l, list)
        return l

    def shared_step(
        self,
        prefix: Literal["train", "test", "val"],
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> STEP_OUTPUT:
        metrics = self.make_step_metrics(*batch)
        self.get_step_metrics_list(prefix).append(metrics)
        return metrics

    # --- Lightning APIs ---

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int
    ) -> STEP_OUTPUT:
        return self.shared_step("train", batch)

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int
    ) -> STEP_OUTPUT:
        return self.shared_step("val", batch)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_index: int
    ) -> STEP_OUTPUT:
        return self.shared_step("test", batch)

    def predict_step(
        self, batch: torch.Tensor | np.ndarray, batch_index: int
    ) -> torch.Tensor:
        return self(batch)

    def on_train_epoch_end(self) -> None:
        """
        This method call last, see details :
        Hooks order: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks
        Log default: https://lightning.ai/docs/pytorch/stable/extensions/logging.html
        Model checkpoint( see: save_on_train_epoch_end): https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint
        """
        # Compose both train and val and save only one time.
        train_metrics_list = self.get_step_metrics_list("train")
        val_metrics_list = self.get_step_metrics_list("val")

        metrics = {}
        metrics.update(self.make_epoch_metrics("train", train_metrics_list))
        metrics.update(self.make_epoch_metrics("val", val_metrics_list))
        self.log_dict(metrics, logger=True)

        train_metrics_list.clear()
        val_metrics_list.clear()

    # def on_validation_epoch_end(self) -> None:
    #     self.shared_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        test_metrics_list = self.get_step_metrics_list("test")
        self.log_dict(self.make_epoch_metrics("test", test_metrics_list), logger=True)
        test_metrics_list.clear()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.AdamW(self.parameters())
