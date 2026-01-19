import time
import torch
import pandas as pd
from collections import deque
from models.dendritic_cnn import DendriticCNN

model = DendriticCNN()
model.load_state_dict(torch.load("dendritic_model.pt"))
model.eval()

window = deque(maxlen=20)
df = pd.read_csv("data/raw/simulated_vehicle_data.csv")

for _, row in df.iterrows():
    sensors = row[1:7].values
    window.append(sensors)

    if len(window) == 20:
        x = torch.tensor([window], dtype=torch.float32)
        start = time.time()
        pred = model(x).argmax().item()
        latency = (time.time() - start) * 1000

        print(f"Prediction: {pred} | Latency: {latency:.2f} ms")

    time.sleep(0.01)
