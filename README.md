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
  ![Stop Sign](data/Test/00111.png)

- **Yield Sign**:  
  ![Yield Sign](data/Test/00120.png)

- **Speed Limit Sign**:  
  ![Speed Limit Sign](data/Test/00122.png)

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
