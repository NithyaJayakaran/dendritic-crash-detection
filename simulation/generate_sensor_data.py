import numpy as np
import pandas as pd

np.random.seed(42)

fs = 100  # Hz
duration = 30  # seconds
samples = fs * duration
time = np.linspace(0, duration, samples)

ax = np.random.normal(0, 0.2, samples)
ay = np.random.normal(0, 0.2, samples)
yaw_rate = np.random.normal(0, 0.05, samples)
wheel_slip = np.random.normal(0.05, 0.02, samples)
brake_pressure = np.random.normal(0.1, 0.05, samples)
vibration = np.random.normal(0.01, 0.005, samples)

label = np.zeros(samples)

# instability
instability_idx = int(12 * fs)
ax[instability_idx:instability_idx+200] += 1.0
yaw_rate[instability_idx:instability_idx+200] += 0.5
label[instability_idx:instability_idx+200] = 1

# crash
crash_idx = int(20 * fs)
ax[crash_idx:crash_idx+50] += 5.0
vibration[crash_idx:crash_idx+50] += 3.0
label[crash_idx:crash_idx+50] = 2

df = pd.DataFrame({
    "time": time,
    "ax": ax,
    "ay": ay,
    "yaw_rate": yaw_rate,
    "wheel_slip": wheel_slip,
    "brake_pressure": brake_pressure,
    "vibration": vibration,
    "label": label
})

df.to_csv("data/raw/simulated_vehicle_data.csv", index=False)
print("Saved raw simulated vehicle data.")
