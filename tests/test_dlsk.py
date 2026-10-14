import pytest
import torch

from dlsk import DLSK, DLSKBlock


@pytest.mark.parametrize("shape", [(1, 8, 32, 32), (2, 16, 17, 23), (1, 4, 1, 1)])
def test_output_shape_matches_input(shape):
    x = torch.randn(*shape)
    model = DLSK(shape[1])
    y = model(x)
    assert y.shape == x.shape


def test_backward_pass():
    x = torch.randn(2, 8, 16, 16, requires_grad=True)
    model = DLSK(8)
    model(x).mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_invalid_channels_raise_error():
    with pytest.raises(ValueError):
        DLSK(0)
    with pytest.raises(ValueError):
        DLSK(8)(torch.randn(1, 4, 16, 16))


def test_invalid_input_rank_raises_error():
    with pytest.raises(ValueError):
        DLSK(8)(torch.randn(1, 8, 16))


def test_block_shape():
    x = torch.randn(1, 8, 16, 16)
    assert DLSKBlock(8)(x).shape == x.shape