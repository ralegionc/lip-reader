"""
LipCoordNet -- a two-branch sentence-level lip reading network.

Branch 1 (appearance): 3x 3D-conv + 2x bidirectional GRU over the mouth crops.
Branch 2 (geometry):   1x bidirectional GRU over the 20 lip landmarks.
The two are concatenated per timestep and projected to 28 CTC classes
(blank + 26 letters + space).

Architecture and the pretrained weights come from LipCoordNet by ffeew
(https://github.com/ffeew/LipCoordNet), itself built on LipNet
(Assael et al., 2016). Reproduced here so the project runs standalone.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.init as init


class LipCoordNet(nn.Module):
    def __init__(
        self,
        dropout_p: float = 0.5,
        coord_input_dim: int = 40,
        coord_hidden_dim: int = 128,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.gru1 = nn.GRU(96 * 4 * 8, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        # Geometry branch.
        self.coord_gru = nn.GRU(
            coord_input_dim, coord_hidden_dim, 1, bidirectional=True
        )

        self.FC = nn.Linear(512 + 2 * coord_hidden_dim, 27 + 1)

        self.dropout_p = dropout_p
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)

        self._init()

    def _init(self) -> None:
        for conv in (self.conv1, self.conv2, self.conv3):
            init.kaiming_normal_(conv.weight, nonlinearity="relu")
            init.constant_(conv.bias, 0)

        init.kaiming_normal_(self.FC.weight, nonlinearity="sigmoid")
        init.constant_(self.FC.bias, 0)

        for m in (self.gru1, self.gru2):
            stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
            for i in range(0, 256 * 3, 256):
                init.uniform_(
                    m.weight_ih_l0[i:i + 256],
                    -math.sqrt(3) * stdv, math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0[i:i + 256])
                init.constant_(m.bias_ih_l0[i:i + 256], 0)
                init.uniform_(
                    m.weight_ih_l0_reverse[i:i + 256],
                    -math.sqrt(3) * stdv, math.sqrt(3) * stdv,
                )
                init.orthogonal_(m.weight_hh_l0_reverse[i:i + 256])
                init.constant_(m.bias_ih_l0_reverse[i:i + 256], 0)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, 3, T, 64, 128) float, already scaled to [0, 1]
        coords: (B, T, 20, 2) float
        returns (B, T, 28) unnormalised logits
        """
        # --- appearance branch ---
        x = self.pool1(self.dropout3d(self.relu(self.conv1(x))))
        x = self.pool2(self.dropout3d(self.relu(self.conv2(x))))
        x = self.pool3(self.dropout3d(self.relu(self.conv3(x))))

        # (B, C, T, H, W) -> (T, B, C*H*W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        x = x.view(x.size(0), x.size(1), -1)

        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        x, _ = self.gru1(x)
        x = self.dropout(x)
        x, _ = self.gru2(x)
        x = self.dropout(x)

        # --- geometry branch ---
        self.coord_gru.flatten_parameters()
        coords = coords.permute(1, 0, 2, 3).contiguous()
        coords = coords.view(coords.size(0), coords.size(1), -1)
        coords, _ = self.coord_gru(coords)
        coords = self.dropout(coords)

        # --- fuse ---
        out = self.FC(torch.cat((x, coords), dim=2))
        return out.permute(1, 0, 2).contiguous()
