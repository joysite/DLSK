"""
DLSK: Directional Large Selective Kernel.

A lightweight convolutional module for exploring directional features in
computer-vision tasks involving elongated structures.
"""

from typing import Optional

import torch
import torch.nn as nn


class DLSK(nn.Module):
    """Directional Large Selective Kernel module.

    Args:
        dim: Number of input and output channels. Must be positive.
        attn_kernel_size: Odd kernel size used by the spatial-attention layer.

    Input:
        Tensor with shape (B, C, H, W).

    Output:
        Tensor with shape (B, C, H, W).
    """

    def __init__(self, dim: int, attn_kernel_size: int = 5) -> None:
        super().__init__()

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
        if (
            not isinstance(attn_kernel_size, int)
            or attn_kernel_size <= 0
            or attn_kernel_size % 2 == 0
        ):
            raise ValueError("attn_kernel_size must be a positive odd integer")

        self.dim = dim
        self.conv_local = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=True
        )
        self.conv_vertical = nn.Conv2d(
            dim,
            dim,
            kernel_size=(11, 1),
            padding=(10, 0),
            dilation=2,
            groups=dim,
            bias=True,
        )
        self.conv_horizontal = nn.Conv2d(
            dim,
            dim,
            kernel_size=(1, 11),
            padding=(0, 10),
            dilation=2,
            groups=dim,
            bias=True,
        )
        self.conv_fusion = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv_attn = nn.Conv2d(
            2, 3, kernel_size=attn_kernel_size, padding=attn_kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute direction-aware feature refinement."""
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor (B, C, H, W), got {x.shape}")
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected {self.dim} input channels, got {x.shape[1]}"
            )

        feat_local = self.conv_local(x)
        feat_vertical = self.conv_vertical(feat_local)
        feat_horizontal = self.conv_horizontal(feat_local)

        feat_joint = self.conv_fusion(
            feat_local + feat_vertical + feat_horizontal
        )

        avg_desc = torch.mean(feat_joint, dim=1, keepdim=True)
        max_desc = torch.amax(feat_joint, dim=1, keepdim=True)
        attention_desc = torch.cat((avg_desc, max_desc), dim=1)
        attention = torch.sigmoid(self.conv_attn(attention_desc))

        weight_local = attention[:, 0:1]
        weight_vertical = attention[:, 1:2]
        weight_horizontal = attention[:, 2:3]

        return (
            feat_local * weight_local
            + feat_vertical * weight_vertical
            + feat_horizontal * weight_horizontal
        )


class DLSKBlock(nn.Module):
    """DLSK followed by optional BatchNorm and activation.

    This wrapper is convenient for integrating DLSK into CNN backbones.
    Set norm_layer=None or activation=None to disable either operation.
    """

    def __init__(
        self,
        dim: int,
        norm_layer: Optional[nn.Module] = nn.BatchNorm2d,
        activation: Optional[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        layers = [DLSK(dim)]
        if norm_layer is not None:
            layers.append(norm_layer(dim))
        if activation is not None:
            layers.append(activation(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)