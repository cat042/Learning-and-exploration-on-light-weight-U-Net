"""Archived implementation of the initial lightweight UNet scheme.

This module stores a runnable model definition matching the first design:
- encoder keeps 3 main channels throughout
- each input channel expands with three independent 2x2 kernels
- two rounds of 2x2 depthwise convolution + BN + ReLU6
- regroup back to 3 channels before the final cross-channel linear mix
- decoder concatenation forms 6 channels, regrouped into 3 pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePadConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        total_pad = kernel_size - 1
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        pad_top = total_pad // 2
        pad_bottom = total_pad - pad_top
        self.pad = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class ChannelLinearMix(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(channels, channels))
        self.register_buffer("mask", torch.ones(channels, channels) - torch.eye(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = torch.einsum("ij,bjhw->bihw", self.weights * self.mask, x)
        return x + mixed


class IndependentExpand(nn.Module):
    def __init__(self, in_channels: int, groups: int, group_in_channels: int) -> None:
        super().__init__()
        self.groups = groups
        self.group_in_channels = group_in_channels
        self.expand = SamePadConv2d(
            in_channels,
            in_channels * 3,
            kernel_size=2,
            groups=in_channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        expanded = self.expand(x).reshape(batch_size, self.groups, self.group_in_channels, 3, height, width)
        expanded = expanded.permute(0, 1, 3, 2, 4, 5)
        return expanded.reshape(batch_size, -1, height, width)


class InitialIndependentBlock(nn.Module):
    def __init__(self, group_in_channels: int, groups: int = 3) -> None:
        super().__init__()
        in_channels = groups * group_in_channels
        expanded_channels = in_channels * 3

        self.expand = nn.Sequential(
            IndependentExpand(in_channels, groups=groups, group_in_channels=group_in_channels),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
        )
        self.dw1 = nn.Sequential(
            SamePadConv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=2,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
        )
        self.dw2 = nn.Sequential(
            SamePadConv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=2,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
        )
        self.reduce = nn.Sequential(
            SamePadConv2d(expanded_channels, groups, kernel_size=2, groups=groups, bias=False),
            nn.BatchNorm2d(groups),
            nn.ReLU6(inplace=True),
        )
        self.mix = ChannelLinearMix(channels=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.dw1(x)
        x = self.dw2(x)
        x = self.reduce(x)
        return self.mix(x)


class Down(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), InitialIndependentBlock(group_in_channels=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2)
        self.conv = InitialIndependentBlock(group_in_channels=2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat(
            [
                x2[:, 0:1],
                x1[:, 0:1],
                x2[:, 1:2],
                x1[:, 1:2],
                x2[:, 2:3],
                x1[:, 2:3],
            ],
            dim=1,
        )
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("This archived initial UNet expects exactly 3 input channels.")

        self.base_channels = base_channels
        self.inc = InitialIndependentBlock(group_in_channels=1)
        self.down1 = Down()
        self.down2 = Down()
        self.down3 = Down()
        self.down4 = Down()
        self.up1 = Up()
        self.up2 = Up()
        self.up3 = Up()
        self.up4 = Up()
        self.outc = OutConv(3, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


if __name__ == "__main__":
    model = UNet()
    sample = torch.randn(1, 3, 256, 256)
    output = model(sample)
    print("output_shape=", tuple(output.shape))
