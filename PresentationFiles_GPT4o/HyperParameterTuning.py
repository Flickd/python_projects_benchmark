net = Net(config["l1"], config["l2"])

device = "cpu"
# Check if CUDA is available and set the device to GPU if possible
if torch.cuda.is_available():
    device = "cuda:0"
    # Use DataParallel to utilize multiple GPUs if available
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
# Move the network to the selected device
net.to(device)

criterion = nn.CrossEntropyLoss()
# Initialize the optimizer with Stochastic Gradient Descent
optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

# Load the training and testing datasets
trainset, testset = load_data(data_dir)

# Split the training set into training and validation subsets
test_abs = int(len(trainset) * 0.8)
train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])

# Create data loaders for training and validation
trainloader = torch.utils.data.DataLoader(
    train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
)
valloader = torch.utils.data.DataLoader(
    val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
)

# Training loop for multiple epochs
for epoch in range(start_epoch, 10):
    running_loss = 0.0
    epoch_steps = 0
    # Iterate over the training data
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # Move inputs and labels to the selected device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass, backward pass, and optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate loss and count steps
        running_loss += loss.item()
        epoch_steps += 1
        # Print statistics every 2000 mini-batches
        if i % 2000 == 1999:
            print(
                "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps)
            )
            running_loss = 0.0

    # Validation phase
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    # Iterate over the validation data
    for i, data in enumerate(valloader, 0):
        with torch.no_grad():
            inputs, labels = data
            # Move inputs and labels to the selected device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            # Calculate total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Accumulate validation loss
            loss = criterion(outputs, labels)
            val_loss += loss.cpu().numpy()
            val_steps += 1

print("Finished Training")
