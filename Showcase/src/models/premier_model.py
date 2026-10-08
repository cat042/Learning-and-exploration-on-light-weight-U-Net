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


class CrossGroupLinearMix(nn.Module):
    def __init__(self, groups: int, group_in_channels: int, scales_per_channel: int) -> None:
        super().__init__()
        channels_per_group = group_in_channels * scales_per_channel
        channels = groups * channels_per_group

        mask = torch.zeros(channels, channels)
        for out_group in range(groups):
            out_group_start = out_group * channels_per_group
            for out_scale in range(scales_per_channel):
                out_start = out_group_start + out_scale * group_in_channels
                out_end = out_start + group_in_channels

                for in_group in range(groups):
                    if in_group == out_group:
                        continue

                    in_group_start = in_group * channels_per_group
                    in_start = in_group_start + out_scale * group_in_channels
                    in_end = in_start + group_in_channels
                    mask[out_start:out_end, in_start:in_end] = 1.0
        self.weights = nn.Parameter(torch.zeros(channels, channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = torch.einsum("ij,bjhw->bihw", self.weights * self.mask, x)
        return x + mixed + self.bias.view(1, -1, 1, 1)


class ChannelLinearMix(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(channels, channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("mask", torch.ones(channels, channels) - torch.eye(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = torch.einsum("ij,bjhw->bihw", self.weights * self.mask, x)
        return x + mixed + self.bias.view(1, -1, 1, 1)


class MultiKernelExpand(nn.Module):
    def __init__(self, in_channels: int, groups: int, group_in_channels: int) -> None:
        super().__init__()
        self.groups = groups
        self.group_in_channels = group_in_channels
        self.group_branches = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SamePadConv2d(1, 1, kernel_size=1, bias=False),
                        SamePadConv2d(1, 1, kernel_size=2, bias=False),
                        SamePadConv2d(1, 1, kernel_size=3, bias=False),
                    ]
                )
                for _ in range(groups)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        x = x.reshape(batch_size, self.groups, self.group_in_channels, height, width)
        group_outputs = []
        for group_index, group_branches in enumerate(self.group_branches):
            x_group = x[:, group_index]
            channel_outputs = []
            for channel_index in range(self.group_in_channels):
                x_channel = x_group[:, channel_index : channel_index + 1]
                scale_outputs = [branch(x_channel) for branch in group_branches]
                channel_outputs.append(torch.cat(scale_outputs, dim=1))
            group_outputs.append(torch.cat(channel_outputs, dim=1))

        expanded = torch.stack(group_outputs, dim=1)
        expanded = expanded.reshape(batch_size, self.groups, self.group_in_channels, 3, height, width)
        expanded = expanded.permute(0, 1, 3, 2, 4, 5)
        return expanded.reshape(batch_size, -1, height, width)


class IndependentBlock(nn.Module):
    def __init__(self, group_in_channels: int, groups: int = 3) -> None:
        super().__init__()
        in_channels = groups * group_in_channels
        scales_per_channel = 3
        channels_per_group = group_in_channels * scales_per_channel
        expanded_channels = groups * channels_per_group

        self.expand = nn.Sequential(
            MultiKernelExpand(in_channels, groups=groups, group_in_channels=group_in_channels),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=True),
        )
        self.mix0 = CrossGroupLinearMix(
            groups=groups,
            group_in_channels=group_in_channels,
            scales_per_channel=scales_per_channel,
        )
        self.dw1 = nn.Sequential(
            SamePadConv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=3,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=True),
        )
        self.mix1 = CrossGroupLinearMix(
            groups=groups,
            group_in_channels=group_in_channels,
            scales_per_channel=scales_per_channel,
        )
        self.dw2 = nn.Sequential(
            SamePadConv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=3,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=True),
        )
        self.mix2 = CrossGroupLinearMix(
            groups=groups,
            group_in_channels=group_in_channels,
            scales_per_channel=scales_per_channel,
        )
        self.reduce = nn.Sequential(
            SamePadConv2d(expanded_channels, groups, kernel_size=2, groups=groups, bias=False),
            nn.BatchNorm2d(groups),
            nn.ReLU(inplace=True),
        )
        self.final_mix = ChannelLinearMix(channels=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.mix0(x)
        x = self.dw1(x)
        x = self.mix1(x)
        x = self.dw2(x)
        x = self.mix2(x)
        x = self.reduce(x)
        return self.final_mix(x)


class Down(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), IndependentBlock(group_in_channels=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2)
        self.conv = IndependentBlock(group_in_channels=2)

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
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.conv2(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("This lightweight UNet expects exactly 3 input channels.")

        self.base_channels = base_channels
        self.inc = IndependentBlock(group_in_channels=1)
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