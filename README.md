# Traductor LENSEGUA
<img src="https://raw.githubusercontent.com/achleey/TraductorLENSEGUA/main/src/LENSEGUA.jpeg" width="500" height = 300>

This project aims to develop a comprehensive desktop application for translating Guatemalan Sign Language (LENSEGUA) using computer vision and machine learning. The initial version sets up the framework for real-time video processing and sign language recognition, utilizing computer vision techniques to capture and interpret signs. The application integrates machine learning models to translate detected signs into text and audio, providing an interactive and efficient translation tool. The ultimate goal is to create a robust system for translating Guatemalan Sign Language in real-time, facilitating communication and accessibility.

Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)

## Features

- Real-time Sign Language Translation: Utilizes computer vision and machine learning to detect and translate Guatemalan Sign Language (LENSEGUA) signs in real-time from video input. The system captures live video, processes the frames using OpenCV, and interprets gestures with Mediapipe for accurate sign recognition.
- Real-time Landmark Detection: Uses Mediapipe for detecting and tracking landmarks on hands, face, and body. This information aids in accurate sign language interpretation and provides essential data for translating gestures.
- Integration with Machine Learning Models: Employs Scikit-learn to train classifiers for recognizing static signs. The models are evaluated and optimized to ensure precise identification of signs based on real-time data.
- User Interface: Designed in Figma and implemented using PyQt6, the interface integrates video capture, sign language detection, and translation display. It features interactive elements such as buttons and switches to control various functionalities and manage user preferences.
- Customizable Translation Options: Allows users to toggle features such as the visibility of landmarks and translation modes, adapting the application to different user needs and preferences.
- Text and Audio Output: Provides translation of detected signs into both text and audio. This feature allows users to see the translated sign language and hear its corresponding audio representation, enhancing communication accessibility.
- Open Source Integration: Incorporates elements from [this GitHub repository](https://github.com/computervisioneng/sign-language-detection-python), adapting and customizing existing solutions to fit the specific requirements of LENSEGUA. This includes optimizing data handling, feature extraction, and real-time classification techniques.
