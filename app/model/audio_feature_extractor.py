import torch.nn as nn


class AudioFeatureExtractor(nn.Module):
    def __init__(self, dropout=0.1, input_channels=1, output_channels=16):
        """
        2D CNN for audio feature extraction, processing slices of the spectrogram.
        Args:
            input_channels (int): Number of input channels (default 1).
            output_channels (int): Number of output features (default 16).
            dropout (float): Dropout probability 

        """
        super(AudioFeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels,
                               (3, 3), padding=(1, 0))
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool2d((1, 3))

        self.conv2 = nn.Conv2d(
            output_channels, output_channels, (3, 3), padding=(1, 0))
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool2d((1, 3))

        self.conv3 = nn.Conv2d(output_channels, output_channels, (1, 8))
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, spectrogram):
        """
        Forward pass for audio feature extraction.
        Args:
            spectrogram (torch.Tensor): Input spectrogram of shape (batch_size, T, 81), where T is the number of video frames.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, T, 16).
        """
        # Reshape the input to match the expected input shape for Conv2d
        # Add channel dimension. Shape: (batch_size, 1, T, 81)
        spectrogram = spectrogram.unsqueeze(1)

        y = self.conv1(spectrogram)
        y = self.elu1(y)
        y = self.dropout1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.elu2(y)
        y = self.dropout2(y)
        y = self.pool2(y)
        y = self.conv3(y)
        y = self.elu3(y)

        # Remove the frequency dimension (which should now be 1)
        y = y.squeeze(-1)  # Shape: (batch_size, 16, T)

        # Permute dimensions to have shape (batch_size, T, 16)
        y = y.permute(0, 2, 1)  # Shape: (batch_size, T, 16)

        return y
