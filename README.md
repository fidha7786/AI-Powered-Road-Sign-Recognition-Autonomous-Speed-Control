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
    python main.py --data --data_dir data

```
### Exploratory Data Analysis (EDA)

To generate visualizations and understand the dataset:
```
    python main.py --eda --data_dir data --output_dir outputs

```
### Model Training

To train the CNN model on the preprocessed data:
```
     python main.py --training --data_dir data --model_dir models --epochs 20 --      batch_size 64 --learning_rate 0.001
```
### Model Evaluation

To evaluate the trained model:
```
   python main.py --evaluation --data_dir data --model_dir models --output_dir      outputs
```
### Inference

To make predictions on new images:
```
     python main.py --inference --model_dir models --image_path path/to/image.jpg

```
### IStreamlit Application

To launch the Streamlit application for interactive traffic sign recognition:

```
    streamlit run scripts/streamlit_app.py -- --model_path models/best_model.h5
```
### live Detection Using camera
can customize thresh0ld and camera:
```
python main.py --live --camer 1 --threshold 0.8
```
for keeping delay while detecting :
```
python main.py --live --cooldown 2
```
### Notes

Image Paths: Ensure to replace data/Train/00000.png, etc., with actual paths to your images.


Repository URL: Replace <repository-url> with the actual URL of your GitHub repository.


License: Ensure you have a LICENSE file if you include a license section.


This README.md provides a comprehensive overview of your project, making it easy for users to understand its purpose and how to use it effectively. Let me know if you need further adjustments or additions!
