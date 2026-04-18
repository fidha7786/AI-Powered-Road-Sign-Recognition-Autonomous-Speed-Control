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
