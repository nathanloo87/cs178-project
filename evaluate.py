from imports import *
from cnn import SVHNCNN
from load_dataset import test_loader

""" THIS ONE IS STILL JANK GOTTA WORK ON THIS ONE """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SVHNCNN().to(device)
model.load_state_dict(torch.load("svhn_cnn.pth"))  # this one loads trained weights
model.eval()

# evaluate accuracy
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
