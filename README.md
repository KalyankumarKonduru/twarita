<p align="center">
  <img src="https://github.com/user-attachments/assets/8f420cfc-e4cf-4801-a2e0-d5c65fd9045f" />
</p>

# AI-Powered Risk Assessment and Adaptive Decision-Making System

<p align="center">
  <img src="https://github.com/user-attachments/assets/7f1fc07a-4146-4da1-a783-d759ee05ff26" alt="0323" />
</p>

## Project Overview
TWARITA is an AI-powered system designed to enhance vehicle safety by combining object detection with real-time collision risk assessment and adaptive decision-making. Developed as a robust solution for dynamic driving conditions, TWARITA leverages advanced AI techniques to predict and respond to potential threats. Key components include:

- **Risk Assessment Algorithm** for predicting collisions based on sensor data.
- **Adaptive Warning Mechanisms** to alert drivers or trigger vehicle responses.
- **Scenario-Based Simulations** to test performance in diverse conditions.

The system processes sensor inputs, assesses risks, and provides actionable outputs in real-time, making it suitable for integration into modern vehicle platforms.

## Key Features
- **Risk Assessment**: Predicts collisions using object distance, velocity, trajectory, and vehicle speed, powered by AI models like Reinforcement Learning or Decision Trees.
- **Adaptive Alerts**: Delivers smart warnings (audio-visual signals, dashboard notifications, or haptic feedback) based on real-time threat analysis.
- **Real-Time Decision-Making**: Simulates vehicle assistance features such as automatic braking and lane departure warnings.
- **Scenario Testing**: Evaluates performance in challenging scenarios like merging lanes, sharp turns, and highway driving in fog or rain.
- **Scalable Design**: Built for deployment on edge devices and integration with autonomous driving systems.

## Vision
TWARITA aims to evolve into a **Software-as-a-Service (SaaS)** platform for vehicle manufacturers, offering an end-to-end solution for risk assessment and adaptive decision-making. Future enhancements include:

- **Multi-Sensor Integration**: Combining camera, LIDAR, and radar data for comprehensive analysis.
- **Ethical Data Practices**: Collecting real-world driving data with user consent for continuous improvement.
- **Global Deployment**: Validating the system across diverse geographic and weather conditions.

## Problem Statement
Detecting objects alone is not sufficient—a system must assess collision risk and trigger warnings or adaptive vehicle responses in real-time. TWARITA addresses this by:
- Developing an AI-based risk assessment model to classify potential threats using sensor data.
- Implementing adaptive warning mechanisms for driver and vehicle response.
- Simulating and testing real-time decision-making in varied driving scenarios.

## Repository Structure
```
TWARITA/
├── twarita/
│   ├── risk_assesment_system.py              # Main script for risk assessment and decision-making
│   ├── output/
│   │   ├── output.mp4  # Output simulations
    ├── input/                   # Directory for scenario-based testing files
        ├── video1.mp4                # Script to train the risk assessment                # Input video or sensor data for testing
└── README.md
```

## Setup Instructions
### Prerequisites
- **Python 3.8+**
- **Operating System**: Tested on macOS; compatible with Windows and Ubuntu 20.04 with minor tweaks.
- **Hardware**: GPU recommended for training and inference (e.g., NVIDIA GPU with CUDA support).

### Installation
#### Clone the Repository
```bash
git clone https://github.com/username/twarita.git
cd twarita
```

#### Set Up a Virtual Environment (Recommended)
```bash
python3 -m venv twarita.env
source twarita.env/bin/activate  # On Windows: twarita.env\Scripts\activate
```

#### Install Dependencies
```bash
pip install opencv-python numpy pygame ultralytics scipy filterpy matplotlib tensorflow
```

#### Prepare the Input Data
- Place your input video or sensor data as `video1.mp4` in the `TWARITA/` directory.
- Modify the input path in `inference.py` if needed.

## Usage Instructions
### Running the Inference Script
Navigate to the TWARITA directory:
```bash
cd twarita
```

#### Run the script:
```bash
python3 inference.py
```

### Viewing the Output
The output simulation will be saved as:
```bash
output/output_simulation_<timestamp>.mp4
```

## Results and Performance
### Output Simulation
- **Left Side**: Raw input with scenario details (e.g., lane merge, foggy highway).
- **Right Side**: Processed output with risk indicators (e.g., red alerts for high-risk objects) and adaptive responses.
<p align="center">
<img width="683" alt="Screenshot 2025-03-24 at 10 05 13 PM" src="https://github.com/user-attachments/assets/f3319eee-79ed-4618-8d8a-6349034c8fed" />
</p>

### Performance Metrics
- **Risk Prediction Accuracy**: Precision and recall metrics TBD post-training.
- **Real-Time Performance**: Target of <40ms per frame for edge compatibility.

## Future Work
- **Model Enhancement**: Integrate Bayesian Networks or Reinforcement Learning for improved risk prediction.
- **Sensor Fusion**: Combine camera, LIDAR, and radar inputs for robust analysis.
- **Scenario Expansion**: Add urban intersections, pedestrian-heavy zones, and extreme weather.
- **Response Simulation**: Implement advanced vehicle controls like adaptive cruise control.
- **Edge Optimization**: Deploy on devices like NVIDIA Jetson for real-world testing.

---
TWARITA is a forward-thinking AI system that enhances vehicle safety through risk assessment and adaptive decision-making. Future iterations will focus on multi-sensor integration, real-time performance, and seamless adoption in autonomous driving ecosystems.
