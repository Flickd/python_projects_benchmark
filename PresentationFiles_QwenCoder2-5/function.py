import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def test_autoencoder_braindata(net, testset: TensorDataset, device="cpu"):
    """
    Evaluates the performance of an autoencoder on a given test dataset.

    Args:
        net (nn.Module): The autoencoder model to be evaluated.
        testset (TensorDataset): The test dataset containing input and target tensors.
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        tuple: A tuple containing the average loss, test encodings as a numpy array, and test labels as a numpy array.
    """

    # Determine the appropriate device based on CUDA availability
    if torch.cuda.is_available():
        device = "cuda:0"
        # If multiple GPUs are available, use DataParallel for parallel processing
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)

    # Move the model to the selected device
    net.to(device)

    # Create a DataLoader for efficient batching and data loading
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    # Initialize loss function and variables to store cumulative loss and loss list
    loss_function = nn.MSELoss()
    acc_loss = 0.0
    loss_list = []

    # Set the model to evaluation mode
    net.eval()

    # Disable gradient computation during inference for memory efficiency
    with torch.no_grad():
        for data in testloader:
            inputs = data[0].to(device)
            _, output = net(inputs)  # Forward pass through the autoencoder
            loss = loss_function(output, inputs)  # Compute the reconstruction loss
            acc_loss += loss.item()  # Accumulate the loss
            loss_list.append(loss.item())  # Append the loss to the list

    # Get encodings of the test data and move them back to CPU
    test_encodings_tensor, _ = net(testset.tensors[0].to(device))
    test_encodings = test_encodings_tensor.cpu().detach().numpy()

    # Extract labels from the test dataset
    test_labels = testset.tensors[1].numpy()

    # Print the shape of the test dataset and average loss
    print("Testset:", testset.tensors[0].shape)
    print("Test Set loss: {}".format(acc_loss / len(testset)))

    # Return the accumulated loss, encodings, and labels
    return acc_loss, test_encodings, test_labels
