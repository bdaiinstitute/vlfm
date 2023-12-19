#!/usr/bin/env python3
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.


# Adapted from:
# https://github.com/facebookresearch/habitat-lab/blob/main/habitat-baselines/habitat_baselines/rl/ddppo/policy/resnet.py
# This is a filtered down version that only support ResNet-18

from typing import List, Optional, Type

from torch import Tensor
from torch import nn as nn
from torch.nn.modules.conv import Conv2d


class BasicBlock(nn.Module):
    expansion = 1
    resneXt = False

    def __init__(
        self,
        inplanes: int,
        planes: int,
        ngroups: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        cardinality: int = 1,
    ) -> None:
        super(BasicBlock, self).__init__()
        self.convs = nn.Sequential(
            conv3x3(inplanes, planes, stride, groups=cardinality),
            nn.GroupNorm(ngroups, planes),
            nn.ReLU(True),
            conv3x3(planes, planes, groups=cardinality),
            nn.GroupNorm(ngroups, planes),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.convs(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        groups=groups,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_planes: int,
        ngroups: int,
        block: Type[BasicBlock],
        layers: List[int],
        cardinality: int = 1,
    ) -> None:
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(ngroups, base_planes),
            nn.ReLU(True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cardinality = cardinality

        self.inplanes = base_planes
        if block.resneXt:
            base_planes *= 2

        self.layer1 = self._make_layer(block, ngroups, base_planes, layers[0])  # type: ignore
        self.layer2 = self._make_layer(block, ngroups, base_planes * 2, layers[1], stride=2)  # type: ignore
        self.layer3 = self._make_layer(block, ngroups, base_planes * 2 * 2, layers[2], stride=2)  # type: ignore
        self.layer4 = self._make_layer(block, ngroups, base_planes * 2 * 2 * 2, layers[3], stride=2)  # type: ignore

        self.final_channels = self.inplanes
        self.final_spatial_compress = 1.0 / (2**5)

    def _make_layer(
        self,
        block: BasicBlock,
        ngroups: int,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.GroupNorm(ngroups, planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                ngroups,
                stride,
                downsample,
                cardinality=self.cardinality,
            )
        )
        self.inplanes = planes * block.expansion
        for _i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ngroups))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18(in_channels: int, base_planes: int, ngroups: int) -> ResNet:
    model = ResNet(in_channels, base_planes, ngroups, BasicBlock, [2, 2, 2, 2])

    return model
