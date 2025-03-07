import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Autoencoder(nn.Module):
    """
    A simple autoencoder model implemented using PyTorch.

    This autoencoder consists of an encoder and a decoder. The encoder compresses
    the input data into a lower-dimensional embedding, and the decoder reconstructs
    the data from this embedding.

    Attributes:
        enc_l1 (nn.Linear): First layer of the encoder.
        enc_l2 (nn.Linear): Second layer of the encoder.
        enc_l3 (nn.Linear): Third layer of the encoder, outputs the embedding.
        dec_l1 (nn.Linear): First layer of the decoder.
        dec_l2 (nn.Linear): Second layer of the decoder.
        dec_l3 (nn.Linear): Third layer of the decoder, outputs the reconstructed data.
        relu (nn.ReLU): ReLU activation function used in both encoder and decoder.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim1,
        hidden_dim2,
        embedding_dim,
    ):
        """
        Initializes the Autoencoder with specified dimensions.

        Args:
            input_dim (int): The dimension of the input data.
            hidden_dim1 (int): The dimension of the first hidden layer in the encoder.
            hidden_dim2 (int): The dimension of the second hidden layer in the encoder.
            embedding_dim (int): The dimension of the embedding layer.
        """
        super().__init__()
        self.relu = nn.ReLU()
        self.enc_l1 = nn.Linear(input_dim, hidden_dim1)
        self.enc_l2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.enc_l3 = nn.Linear(hidden_dim2, embedding_dim)

        self.dec_l1 = nn.Linear(embedding_dim, hidden_dim2)
        self.dec_l2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.dec_l3 = nn.Linear(hidden_dim1, input_dim)

    def encoder(self, x):
        """
        Encodes the input data into a lower-dimensional embedding.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The encoded embedding.
        """
        x = self.relu(self.enc_l1(x))
        x = self.relu(self.enc_l2(x))
        x = self.enc_l3(x)
        return x

    def decoder(self, x):
        """
        Decodes the embedding back into the original data dimension.

        Args:
            x (torch.Tensor): The encoded embedding.

        Returns:
            torch.Tensor: The reconstructed data.
        """
        x = self.relu(self.dec_l1(x))
        x = self.relu(self.dec_l2(x))
        x = torch.sigmoid(self.dec_l3(x))
        return x

    def forward(self, x):
        """
        Defines the forward pass of the autoencoder.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple: A tuple containing the encoded embedding and the reconstructed data.
        """
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        return encoding, decoding
