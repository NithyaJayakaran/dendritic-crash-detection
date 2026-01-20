import torch
import numpy as np
from models.dendritic_cnn import DendriticCNN

X = np.load("data/processed/windowed_data.npz")["X"]
y = np.load("data/processed/windowed_data.npz")["y"]

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

model = DendriticCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    optimizer.zero_grad()
    preds = model(X)
    loss = loss_fn(preds, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss {loss.item():.4f}")

torch.save(model.state_dict(), "dendritic_model.pt")
