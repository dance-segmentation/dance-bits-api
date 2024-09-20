import torch
import torch.nn as nn

from .audio_feature_extractor import AudioFeatureExtractor
from .temporal_conv_net import TemporalConvNet
from .visual_feature_extractor import VisualFeatureExtractor


class SegmentationModel(nn.Module):
    def __init__(self, num_bone_vectors=35, input_dims=2, visual_output_dim=35, audio_output_dim=16, tcn_channels=[128]*9):
        super(SegmentationModel, self).__init__()
        # Visual and Audio Feature Extractors
        self.visual_extractor = VisualFeatureExtractor(
            num_bone_vectors, input_dims, visual_output_dim)
        self.audio_extractor = AudioFeatureExtractor(
            output_channels=audio_output_dim)
        # Temporal Convolutional Network
        combined_input_size = visual_output_dim + audio_output_dim
        self.tcn = TemporalConvNet(combined_input_size, tcn_channels)
        # Output layer. Returns one value per frame
        self.fc_out = nn.Linear(tcn_channels[-1], 1)
        self.sigmoid = nn.Sigmoid()  # Binary classification for segmentation point

    def forward(self, visual_input, audio_input):
        # Extract visual features of shape (batch_size, num_frames, num_bone_vectors). num_bone_vectors = 35
        visual_features = self.visual_extractor(visual_input)
        # Extract audio features of shape (batch_size, num_frames, num_audio_features). num_audio_features = 16
        audio_features = self.audio_extractor(audio_input)
        # Combine visual and audio features along the feature dimension to get shape (batch_size, num_frames, 35 + 16)
        combined_features = torch.cat((visual_features, audio_features), dim=2)
        # Pass through TCN to get output of shape (batch_size, num_frames, 128)
        tcn_out = self.tcn(combined_features)
        # Final output layer to get shape (batch_size, num_frames, 1)
        out = self.fc_out(tcn_out)
        return self.sigmoid(out)  # Segmentation probability
