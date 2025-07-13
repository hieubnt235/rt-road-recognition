import numpy as np
import pytest
import torch

from datasets.camvid.meta import classes
from models.deeplabv3p import DeepLabV3Plus, DeepLabV3PlusConfig
from loguru import logger


@pytest.fixture
def config():
    return DeepLabV3PlusConfig(
        label2class={i: c for i, c in enumerate(classes)},
        classes=len(classes),
        strict_input_size=False,
    )


@pytest.fixture()
def model(config):
    return DeepLabV3Plus(config).eval()


@pytest.fixture()
def strict_model(config):
    m = DeepLabV3Plus(config).eval()
    m.strict_input = True
    return m


def test_config(config):
    schema = config.model_dump()
    assert "input_size" not in schema


@pytest.mark.parametrize("input_type", [np.ndarray, torch.Tensor])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("is_batched", [True, False])
@pytest.mark.parametrize("channel_order", ["HWC", "CHW"])
def test_forward_all_valid_cases(model, input_type, dtype, is_batched, channel_order):
    h, w = 64, 64
    if channel_order == "HWC":
        base_shape = (h, w, 3)
    else:  # CHW
        base_shape = (3, h, w)

    shape = (4, *base_shape) if is_batched else base_shape
    batch_size = shape[0] if is_batched else 1

    if dtype == torch.uint8:
        numpy_input = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    else:  # float32
        numpy_input = np.random.rand(*shape).astype(np.float32)

    if input_type is torch.Tensor:
        input_data = torch.from_numpy(numpy_input)
    else:
        input_data = numpy_input

    output = model(input_data)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (
        batch_size,
        model.num_classes,
        h,
        w,
    )  # Output shape is always B, C, H, W
    assert output.device == model.device


def test_strict_input(strict_model):
    correct_size = list(strict_model.input_size + (3,))
    correct_size_input = np.random.rand(*correct_size).astype(np.float32)
    try:
        strict_model(correct_size_input)
    except Exception as e:
        pytest.fail(
            f"forward() raised ValueError unexpectedly in strict mode with correct input size"
        )
        logger.exception(e)

    wrong_size = [2 * s for s in correct_size]
    wrong_size_input = np.random.rand(*wrong_size).astype(np.float32)
    with pytest.raises(ValueError, match="Invalid shape"):
        strict_model(wrong_size_input)

    wrong_size_input2 = np.random.rand(wrong_size[0], wrong_size[1], 3).astype(
        np.float32
    )
    with pytest.raises(ValueError, match="Model is in strict input mode"):
        strict_model(wrong_size_input2)

    wrong_size_input3 = np.random.rand(3, wrong_size[0], wrong_size[1]).astype(
        np.float32
    )
    with pytest.raises(ValueError, match="Model is in strict input mode"):
        strict_model(wrong_size_input2)


def test_invalid_dtype_raises_error(model):
    """
    Tests that an unsupported dtype like int32 raises a ValueError.
    """
    invalid_input = np.random.randint(0, 100, size=(64, 64, 3), dtype=np.int32)

    with pytest.raises(ValueError, match="Input dtype must be float or uint8"):
        model.forward(invalid_input)


def test_train_step(model):
    n, h, w = 5, 32, 32
    ip = torch.rand((n, 3, h, w))
    target = torch.randint(0, model.num_classes, (n, h, w), dtype=torch.int64)
    loss = model.training_step((ip, target), 0)
    assert isinstance(loss, torch.Tensor) or (
        isinstance(loss, dict) and isinstance(loss.get("loss", None), torch.Tensor)
    )
