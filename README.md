# LENSEGUA translator
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
- Open Source Integration: Incorporates elements from [this GitHub repository](https://github.com/computervisioneng/sign-language-detector-python), adapting and customizing existing solutions to fit the specific requirements of LENSEGUA. This includes optimizing data handling, feature extraction, and real-time classification techniques.

## System requirements

This project was developed in a Macbook Pro (Apple Silicon M1 Pro Chip) with macOS Sonoma 14.5. The following software is needed:

- Python 3.10
- PyCharm 2024.1 (Professional Edition). Another IDE can be utilized as long as the migration process is done correctly.

## Repository Structure

Here's an overview of the project's file structure:

- **`Baseline`**: Contains base files sourced from [this GitHub repository](https://github.com/computervisioneng/sign-language-detector-python), which were initially used for testing the project and later adapted to meet specific project needs. Here’s a breakdown of the contents:
    - **`collect_imgs.py`**: Facilitates the collection of labeled image data.
    - **`create_dataset.py`**: Prepares datasets where hand landmarks need to be extracted from images and stored for machine learning or other analysis.
    - **`train_classifier.py`**: Trains a machine learning model on extracted hand landmark data and saves the trained model for later use.
    - **`inference_classifier.py`**: Continuously captures video frames, processes them to detect hand signs, and displays the recognized sign on the video feed.

- **`Own-Data`** and **`Volunteers-Data`**: This folders are organized into subfolders to categorize the signs into different groups, facilitating data management and optimization. The first one is used for a self portrait type of image colection and the second one is used for a volunteer type of image collection. The subfolders are:
    - **`1Hand`**: Contains data and optimized baseline files for signs performed with one hand.
        - **`collect_imgs_1Hand.py`**: Facilitates the collection of 16 different sets of labeled image data, each set corresponding to a specific class.
        - **`create_dataset_1Hand.py`**: Extracts hand landmark coordinates, pads sequences to ensure consistent length, and stores the processed data and labels.
        - **`train_classifier.py`**: Trains a Random Forest classifier on hand landmark data, evaluates its performance and saves the trained model for future use.
    - **`2Hands`**: Contains data and optimized baseline files for signs performed with two hands.
        - **`collect_imgs_2Hands.py`**: Facilitates the collection of three different sets of labeled image data, each set corresponding to a specific class.
        - **`create_dataset_2Hands.py`**: Extracts hand landmark coordinates, pads sequences to ensure consistent length, and stores the processed data and labels.
        - **`train_classifier.py`**: Trains a Random Forest classifier on hand landmark data, evaluates its performance and saves the trained model for future use.
    - **`HandAndFace`**: Contains data and optimized baseline files for signs performed with one hand and the face.
        - **`collect_imgs_HandAndFace.py`**: Facilitates the collection of four different sets of labeled image data, each set corresponding to a specific class.
        - **`create_dataset_HandAndFace.py`**: Extracts hand and face landmark coordinates, pads sequences to ensure consistent length, and stores the processed data and labels.
        - **`train_classifier.py`**: Trains a Random Forest classifier on hand and face landmark data, evaluates its performance and saves the trained model for future use.
    - **`HandAndBody`**: Contains data and optimized baseline files for signs performed with one hand and the arm.
        - **`collect_imgs_HandAndBody.py`**: Facilitates the collection of one set of labeled image data, corresponding to a specific class.
        - **`create_dataset_2Hands.py`**: Extracts hand and body landmark coordinates, pads sequences to ensure consistent length, and stores the processed data and labels.
        - **`train_classifier.py`**: Trains a Random Forest classifier on hand and body landmark data, evaluates its performance and saves the trained model for future use.
    - DISCLAIMER: the files **`draw_xxx`** are only included in the **`Volunteers-Data`** folder. This script reads images from a specified directory, applies hand landmark detection, and visualizes the results. 
