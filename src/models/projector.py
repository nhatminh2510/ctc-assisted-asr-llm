import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)


class AudioProjector(nn.Module):
    """
    Linear projector matching reference EncoderProjectorCov1d:
    Conv1D downsampling -> ReLU -> Linear -> ReLU -> Linear
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 2048,
        output_dim: int = 4096,
        conv_kernel_size: int = 5,
    ):
        super().__init__()

        # Conv1D for downsampling
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=conv_kernel_size,
            stride=conv_kernel_size,
            padding=0,
        )

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu2 = nn.ReLU()

        logger.info(
            f"Initialized AudioProjector: {input_dim} -> {hidden_dim} -> {output_dim}, "
            f"Conv kernel={conv_kernel_size}"
        )

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        if audio_features.dim() != 3:
            raise ValueError(
                f"audio_features must be 3D (batch, seq, feat), got shape {audio_features.shape}"
            )

        # Conv1D downsampling
        x = audio_features.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        # FFN with ReLU (matching reference EncoderProjectorCov1d)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)

        return x
