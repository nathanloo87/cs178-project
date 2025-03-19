from imports import * 
import torch.nn.functional as F
import torch

class SVHNCNN(nn.Module):
    def __init__(self):
        super(SVHNCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Compute correct input size dynamically
        self._to_linear = None  # Placeholder
        self._compute_feature_size()

        self.fc1 = nn.Linear(self._to_linear, 512)  # Dynamically set fc1 input size
        self.fc2 = nn.Linear(512, 10)

    def _compute_feature_size(self):
        """Helper function to determine the size of the feature map before the FC layer."""
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 32, 32)  # Simulate a single 32x32 image
            x = self.pool(F.relu(self.conv1(sample_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            self._to_linear = x.view(1, -1).shape[1]  # Get the flattened size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
