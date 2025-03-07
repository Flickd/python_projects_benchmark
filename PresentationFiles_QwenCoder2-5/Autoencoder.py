import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Autoencoder(nn.Module):
    """
    A simple autoencoder model for dimensionality reduction and data reconstruction.

    This class defines an autoencoder with three encoder layers and three decoder layers. The encoder
    reduces the input dimension to a lower embedding space, and the decoder reconstructs the original
    input from this embedded representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        embedding_dim: int,
    ):
        """
        Initializes the Autoencoder with specified dimensions for each layer.

        Args:
            input_dim (int): The number of features in the input data.
            hidden_dim1 (int): The number of neurons in the first hidden layer.
            hidden_dim2 (int): The number of neurons in the second hidden layer.
            embedding_dim (int): The dimensionality of the embedded space.
        """
        super().__init__()
        self.relu = nn.ReLU()  # ReLU activation function
        self.enc_l1 = nn.Linear(input_dim, hidden_dim1)  # First encoder layer
        self.enc_l2 = nn.Linear(hidden_dim1, hidden_dim2)  # Second encoder layer
        self.enc_l3 = nn.Linear(hidden_dim2, embedding_dim)  # Third encoder layer

        self.dec_l1 = nn.Linear(embedding_dim, hidden_dim2)  # First decoder layer
        self.dec_l2 = nn.Linear(hidden_dim2, hidden_dim1)  # Second decoder layer
        self.dec_l3 = nn.Linear(hidden_dim1, input_dim)  # Third decoder layer

    def encoder(self, x):
        """
        Encodes the input data through a series of linear layers and ReLU activations.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, embedding_dim).
        """
        x = self.relu(self.enc_l1(x))  # Apply ReLU to the first encoder layer
        x = self.relu(self.enc_l2(x))  # Apply ReLU to the second encoder layer
        x = self.enc_l3(x)  # No activation function for the final encoding layer
        return x

    def decoder(self, x):
        """
        Decodes the encoded data back to its original form through a series of linear layers and ReLU activations.

        Args:
            x (torch.Tensor): Encoded tensor of shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Decoded tensor of shape (batch_size, input_dim).
        """
        x = self.relu(self.dec_l1(x))  # Apply ReLU to the first decoder layer
        x = self.relu(self.dec_l2(x))  # Apply ReLU to the second decoder layer
        x = torch.sigmoid(
            self.dec_l3(x)
        )  # Use sigmoid for the final decoding layer to ensure outputs are in [0, 1]
        return x

    def forward(self, x):
        """
        Performs a forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            tuple: A tuple containing the encoded representation and the reconstructed output.
        """
        encoding = self.encoder(x)  # Encode the input
        decoding = self.decoder(encoding)  # Decode the encoded representation
        return encoding, decoding
