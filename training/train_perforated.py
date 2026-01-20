import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.baseline_cnn import BaselineCNN

# Perforated AI import (this comes from their repo)
from perforatedai import initialize_pai

# Load your processed dataset
data = np.load("data/processed/windowed_data.npz")
X = data["X"]   # (N, 20, 6)
y = data["y"]   # (N,)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Train / test split
N = len(X)
split = int(0.8 * N)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

# Define base architecture (same for both runs)
model = BaselineCNN()

# ---- THIS IS THE CRITICAL CALL ----
initialize_pai(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=20,
    save_name="crash_detection",
    use_dendrites=True   # This triggers dendritic optimization
)

print("Training complete. Check PAI/PAI.png for results.")
