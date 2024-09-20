import torch.nn as nn


class VisualFeatureExtractor(nn.Module):
    def __init__(self, num_bone_vectors=35, input_dims=2, output_dims=35):
        """
        Initialize the fully connected network to process bone vectors.

        Args:
            num_bone_vectors (int): The number of bone vectors.
            input_dims (int): The number of input dimensions for each vector (2 for x and y components).
            output_dims (int): The number of output features (default is 35, the same as the number of bone vectors).
        """
        super(VisualFeatureExtractor, self).__init__()

        # Define the fully connected layer (input dimensions are based on the number of bone vectors and input dimensions)
        self.fc = nn.Linear(num_bone_vectors * input_dims, output_dims)
        # Define the activation function. The paper mentions ELU
        self.elu = nn.ELU()

    def forward(self, bone_vectors):
        """
        Forward pass for the visual feature extractor.

        Args:
            bone_vectors (torch.Tensor): Input tensor of bone vectors with shape (batch_size, num_frames, num_bone_vectors * input_dims).

        Returns:
            torch.Tensor: Output feature tensor after the fully connected layer and activation function with shape 
            (batch_size, num_frames, output_dims). output_dims is by default 35 (same as the number of bone vectors).
        """
        # Pass through the fully connected layer
        out = self.fc(bone_vectors)
        # Pass through the activation function
        out = self.elu(out)
        return out
