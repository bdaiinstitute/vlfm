# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import torch
import torch.nn as nn


# modified from CLIP-Adapter -- https://github.com/gaopengcuhk/CLIP-Adapter/blob/main/clip_adapter.py
class Adapter(nn.Module):
    def __init__(
        self,
        c_in: int,
        reduction: int = 16,
        orig_weight: float = 0.1,
        blip: bool = True,
    ) -> None:
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

        self.orig_weight = orig_weight

        self.blip = blip

    def forward(self, x: torch.tensor) -> torch.tensor:
        x2 = self.fc(x)
        if self.blip:
            x2 = x2.reshape(-1, 256)
            x = (x2 / (torch.norm(x2, dim=1).reshape(-1, 1) + 1e-8)).reshape(
                x.shape
            ) * (1 - self.orig_weight) + (
                x.reshape(-1, 256)
                / (torch.norm(x.reshape(-1, 256), dim=1).reshape(-1, 1) + 1e-8)
            ).reshape(
                x.shape
            ) * self.orig_weight
        else:
            x = (x2 / (torch.norm(x2) + 1e-8)).reshape(x.shape) * (
                1 - self.orig_weight
            ) + x * self.orig_weight
        return x
