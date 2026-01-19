import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 16, 3),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3),
            nn.ReLU()
        )
        self.fc = nn.Linear(32 * 16, 3)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
