from imports import * 
from cnn import SVHNCNN
from load_dataset import train_loader, test_loader

print("Loading Model...")  # Debugging print
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = SVHNCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if data is loading correctly
print(f"Train Loader: {len(train_loader)} batches")
print(f"Test Loader: {len(test_loader)} batches")

epochs = 10
for epoch in range(epochs):
    print(f"Starting Epoch {epoch+1}...")  # Debugging print
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Processing batch {batch_idx + 1}/{len(train_loader)}...")  # Debugging print
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} completed. Loss: {running_loss / len(train_loader)}")

    # Validation Step
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy after Epoch {epoch+1}: {accuracy:.2f}%")

print("Training complete!")
