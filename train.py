from imports import * 
from cnn import SVHNCNN
from load_dataset import train_loader, test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SVHNCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train loop with validation
epochs = 10
for epoch in range(epochs):
    model.train()  # now we set the model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # here we do the validation step
    model.eval()  # we can set the model to evaluation mode
    correct, total = 0, 0
    with torch.no_grad():  # we do not need gradient
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Validation Accuracy: {accuracy:.2f}%")

print("Training complete!")
