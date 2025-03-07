net = Net(config["l1"], config["l2"])

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

trainset, testset = load_data(data_dir)

test_abs = int(len(trainset) * 0.8)
train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])

trainloader = torch.utils.data.DataLoader(
    train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
)
valloader = torch.utils.data.DataLoader(
    val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
)

for epoch in range(start_epoch, 10):
    running_loss = 0.0
    epoch_steps = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        epoch_steps += 1
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(
                "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps)
            )
            running_loss = 0.0

    # Validation loss
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    for i, data in enumerate(valloader, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            val_loss += loss.cpu().numpy()
            val_steps += 1

print("Finished Training")
