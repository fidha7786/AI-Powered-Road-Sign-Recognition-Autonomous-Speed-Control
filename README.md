# AI-Powered-Road-Sign-Recognition-Autonomous-Speed-Control
An end-to-end AI/ML and IoT-based system that detects road signs in real time and autonomously adjusts vehicle speed accordingly. This project combines computer vision, deep learning, and embedded systems to simulate intelligent decision-making in smart transportation.

# Overview

This project uses a Convolutional Neural Network (CNN) trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset to classify 43 different road sign categories. The trained model is integrated with a Raspberry Pi and Pi Camera to enable real-time road sign detection in a live environment.

Based on detected speed limit signs, the system dynamically controls a DC motor using GPIO pins and an L293D motor driver, demonstrating an autonomous speed regulation mechanism.

# Tech Stack
Programming Language: Python

Machine Learning: CNN, TensorFlow

Computer Vision: OpenCV

Hardware: Raspberry Pi, Pi Camera, DC Motor, L293D Motor Driver

Deployment/UI: Streamlit

### Example Images
- **Stop Sign**:
- <img width="49" height="46" alt="00014_00000_00002" src="https://github.com/user-attachments/assets/360a6820-d774-4d02-8d0d-bfa50d0cbe7c" />

- **Yield Sign**:
- <img width="29" height="29" alt="00013_00000_00005" src="https://github.com/user-attachments/assets/abe96c05-3b4f-4203-95f3-ce039e5e6676" />

- **Speed Limit Sign**:
- <img width="57" height="59" alt="00001_00000_00006" src="https://github.com/user-attachments/assets/56b58ec4-c4b3-4074-8950-629571ece307" />

## Datasets
The dataset used for this project is the **German Traffic Sign Recognition Benchmark (GTSRB)**. It contains over 50,000 images categorized into 43 classes of traffic signs.

### Dataset Structure
```
data/ 
    ├── Train/ # Contains train images organized by class
    ├── Test/ # Contains test images organized by class
    
```
### CSV Files
- **`Train.csv`**: Contains paths and labels for the training set.
- **`Test.csv`**: Contains paths and labels for the test set.

## Project Structure

```
    traffic-sign-recognition/
    │
    ├── data/
    │   ├── Meta/
    │   ├── Test/
    │   ├── Train/
    │   └── ...
    ├── scripts/
    │   ├── data_preprocessing.py
    │   ├── eda.py
    │   ├── model_training.py
    │   ├── evaluation.py
    │   ├── inference.py
    │   └── streamlit_app.py
    ├── main.py
    ├── requirements.txt
    └── README.md
```

## Setup and Installation
1. **Clone the Repository**:
   ```
        source env/bin/activate  # On Windows use `env\Scripts\activate`
        pip install -r requirements.txt

# Usage

### Data Preprocessing

To preprocess the dataset:
```

```
