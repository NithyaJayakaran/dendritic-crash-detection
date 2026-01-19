import torch
import torch.nn as nn

class DendriticCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # trunk (shared)
        self.trunk = nn.Conv1d(6, 16, 3)

        # dendrites (parallel nonlinear paths)
        self.dendrite1 = nn.Conv1d(16, 16, 3)
        self.dendrite2 = nn.Conv1d(16, 16, 3)

        self.fc = nn.Linear(16 * 14 * 2, 3)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        trunk = torch.relu(self.trunk(x))

        d1 = torch.relu(self.dendrite1(trunk))
        d2 = torch.relu(self.dendrite2(trunk))

        combined = torch.cat([d1, d2], dim=1)
        combined = combined.view(combined.size(0), -1)
        return self.fc(combined)
