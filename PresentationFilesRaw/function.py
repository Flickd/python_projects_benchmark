import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def test_autoencoder_braindata(net, testset: TensorDataset, device="cpu"):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    loss_function = nn.MSELoss()
    acc_loss = 0.0
    loss_list = []
    net.eval()
    with torch.no_grad():
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
