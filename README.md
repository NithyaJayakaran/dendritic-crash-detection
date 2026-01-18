# dendritic-crash-detection
A dendrite-enhanced (rather than traditional neural network), real-time crash and instability detector using IMU and vehicle telemetry data and signals, achieving earlier detection with lower latency and fewer parameters than traditional models.

# Dendritic Crash Detection (CPU Streaming Demo)

## Motivation
Traditional neural networks struggle with fast, interacting sensor signals.
Dendritic models improve early inference by modeling nonlinear signal
interactions across short time windows.

## Sensors Used
- IMU acceleration (ax, ay)
- Yaw rate
- Wheel slip
- Brake pressure
- Chassis vibration

## Input
- 20 timesteps × 6 sensor channels

## Output
- 0 = Normal driving
- 1 = Pre-crash instability
- 2 = Crash detected

## Architecture
- Baseline CNN
- Dendritic-enhanced CNN (Perforated AI)

## Key Feature
- CPU-only real-time streaming inference
- Millisecond-level latency

## Run Order
1. Generate data
2. Train model
3. Run streaming demo
