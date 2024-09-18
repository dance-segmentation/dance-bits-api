import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm


class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_channels=[128]*9, kernel_size=5, dropout=0.1):
        """
        Temporal Convolutional Network (TCN) with residual blocks as described in the paper.
        Each layer has two 1D convolutions, followed by a residual connection.

        Args:
            input_size (int): Number of input channels (visual + audio features).
            num_channels (list of int): Number of channels in each TCN layer.
            kernel_size (int): Kernel size for convolution.
            dropout (float): Dropout rate.
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # Each layer consists of two Conv1D layers + Residual connection
            layers += [ResidualBlock(in_channels, out_channels,
                                     kernel_size, dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for TCN.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames, combined_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_frames, output_dim).
        """
        return self.network(x.transpose(1, 2)).transpose(1, 2)  # Switch back the dimensions


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        """
        A residual block with two Conv1D layers and a residual connection.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size for the convolutions.
            dilation (int): Dilation factor for the convolutions.
            dropout (float): Dropout rate.
        """
        super(ResidualBlock, self).__init__()
        # Apply weight normalization to the convolutions
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=(
            kernel_size - 1) // 2 * dilation, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=(
            kernel_size - 1) // 2 * dilation, dilation=dilation))

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        # 1x1 convolution for matching dimensions if in_channels != out_channels
        self.downsample = nn.Conv1d(
            in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, num_frames).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, num_frames).
        """
        # First convolution
        out = self.conv1(x)
        out = self.elu(out)
        out = self.dropout(out)

        # Second convolution
        out = self.conv2(out)
        out = self.elu(out)
        out = self.dropout(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.elu(out + res)
