import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def test_autoencoder_braindata(net, testset: TensorDataset, device="cpu"):
    """
    Evaluates an autoencoder model on a given test dataset.

    This function tests the performance of an autoencoder neural network on a specified
    test dataset. It computes the mean squared error (MSE) loss between the input and
    reconstructed output. It also returns the accumulated loss, encoded representations
    of the test data, and the corresponding labels.

    Parameters:
    net (torch.nn.Module): The autoencoder model to be tested.
    testset (TensorDataset): A dataset containing input data and labels for testing.
    device (str, optional): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
    tuple: A tuple containing:
        - acc_loss (float): The accumulated loss over the test dataset.
        - test_encodings (numpy.ndarray): The encoded representations of the test inputs.
        - test_labels (numpy.ndarray): The labels corresponding to the test inputs.
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)  # Use multiple GPUs if available
    net.to(device)
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    loss_function = nn.MSELoss()
    acc_loss = 0.0
    loss_list = []
    net.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for data in testloader:
            inputs = data[0].to(device)
            _, output = net(inputs)
            loss = loss_function(output, inputs)
            acc_loss += loss.item()
            loss_list.append(loss.item())

    test_encodings_tensor, _ = net(testset.tensors[0].to(device))
    test_encodings = test_encodings_tensor.cpu().detach().numpy()
    test_labels = testset.tensors[1].numpy()

    print("Testset:", testset.tensors[0].shape)
    print("Test Set loss: {}".format(acc_loss / len(testset)))
    return acc_loss, test_encodings, test_labels
