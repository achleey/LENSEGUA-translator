
![Logo](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F75qpoqrku7bog2weyyno.jpeg)

# LENSEGUA translator 🇬🇹

This project aims to develop a comprehensive desktop application for translating Guatemalan Sign Language (LENSEGUA) using computer vision and machine learning. The initial version sets up the framework for real-time video processing and sign language recognition, utilizing computer vision techniques to capture and interpret signs. The application integrates machine learning models to translate detected signs into text and audio, providing an interactive and efficient translation tool. The ultimate goal is to create a robust system for translating Guatemalan Sign Language in real-time, facilitating communication and accessibility.

## Table of contents

- [Features](#features)
- [Structure](#structure)
- [System requirements](#system-requirements)
- [Acknowledgements](#acknowledgements)
  
## Features

- **Real-time Sign Language Translation:** Detects and translates Guatemalan Sign Language (LENSEGUA) in real-time using OpenCV and Mediapipe for accurate gesture recognition.
- **Real-time Landmark Detection:** Tracks landmarks on hands, face, and body with Mediapipe, aiding in precise sign interpretation.
- **Integration with Machine Learning Models:** Combines Scikit-learn and TensorFlow Lite models for efficient sign recognition, optimized for real-time performance.
- **User Interface:** Developed with PyQt6 and Figma, the interface integrates video capture, sign language detection, and translation display.

    - **Customizable Options:** Lets users toggle features like landmark visibility and translation modes to suit their preferences.

    - **Text and Audio Output:** Provides translated signs in both text and audio for enhanced communication accessibility.
 
## Structure

⚠️ **Disclaimer:**

Some scripts in this project are repeated across multiple sections, as they are adapted for different use cases (e.g., signs with one hand, two hands, face, or body). These scripts may have slight variations in their functionality depending on the specific task. For clarity, the differences and organization of these files are explained in the tables below.

This project is structured as follows:

**01\. Baseline:** Includes original project files used as references for the development of this project. For more details about the original projects refer to [Acknowledgements](#acknowledgements)


- **Static signs:** Contains code for image collection, feature extraction, training a classifier model and real-time prediction of static signs.
        
- **Dynamic signs:** Contains code for video sequence collection, feature extraction, training a classifier model and real-time prediction of dynamic signs.

    | File name | Description                | Folder usage        | Differences| 
    | :-------- | :------------------------- | :------- |:-----------------------|
    | `collect_imgs.py` | Collects labeled image data.| Static signs |-| 
    | `create_dataset.py` | Creates datasets by extracting hand landmarks. | Static signs | -|
    | `collect_sequences.py` | Captures labeled frame sequences and automatically extracts holistic landmarks | Dyamic signs |-|
    | `train_classifier.py` | Trains and evaluates a classifier model for sign language prediction  | Static signs, Dyamic signs |Static signs trains a RandomForest model, Dynamic trains a LSTM model.|
    | `inference_classifier.py` | Detects and predicts sign language in real time.  | Static signs, Dyamic signs |Static signs does simple hand detection and letter classification using a pickle-loaded model, Dynamics uses holistic detection, and TensorFlow for sequence-based predictions.|

**02\. Static signs:** Consists of 2 subfolders:

- **Own-Data**: It consists of 4 subfolders: **`1Hand`**, **`2Hands`**, **`HandAndFace`** and **`HandAndBody`**. It contains code for collecting images, extracting features using MediaPipe solutions based on the body parts used to gesture signs, and training a classifier model using self-portrait images.

- **Volunteers-Data**: It consists of 4 subfolders: **`1Hand`**, **`2Hands`**, **`HandAndFace`** and **`HandAndBody`**. It contains code for extracting features from existing images, plotting landmarks based on the body parts used to gesture signs, and training a classifier model using different volunteers images. 

    | File name | Description                | Folder usage        | Differences| 
    | :-------- | :------------------------- | :----------- |:-----------------------|
    | `collect_imgs_XXX.py` | Collects labeled image data.| Own-Data (all subfolders) |For 1Hand it collects 16 different sets of labeled image data, for 2Hands it collects 3 sets, for HandAndBody it collects 1 set and for HandAndFace it collects 4 sets.| 
    | `create_dataset_XXX.py` | Creates datasets by extracting specific landmarks. | Own-Data, Volunteer-Data (all subfolders for both) | 1Hand and 2Hands use `Hands`, HandAndBody uses `Hands` and `Pose`, HandAndFace uses `Hands`and `FaceMesh` |
    | `draw_XXX.py` | Reads images from a specified directory, applies landmark detection, and plots said landmarks. | Volunteer-Data (all subfolders) | 1Hand and 2Hands plots hand landmarks, HandAndBody plots hand and pose landmarks, HandAndFace plots hand and face landmarks.|
    | `train_classifier.py` | Trains and evaluates a **RandomForest** classifier model for sign language prediction  | Own-Data, Volunteer-Data (all subfolders for both) | The difference consists of the dataset loaded for training.|
      
## System requirements

This project was developed in a Macbook Pro (Apple Silicon M1 Pro Chip) with macOS Sonoma 14.5. The following software is needed:

- Python 3.10
- PyCharm 2024.1 (Professional Edition), or another IDE (as long as the migration process is done correctly).

## Acknowledgements

This section includes resources and references that contributed to the development of the project:

- **Static signs**:
    
    - [Sign Language Detector - GitHub](https://github.com/computervisioneng/sign-language-detector-python)
    - [Sign Language Detector - YouTube](https://www.youtube.com/watch?v=MJCSjXepaAM&t=1072s)

- **Dynamic signs**:

    - [Action Recognition Sign Language Detection - GitHub](https://github.com/nicknochnack/ActionDetectionforSignLanguage)
    - [Action Recognition Sign Language Detection - YouTube](https://www.youtube.com/watch?v=doDUihpj6ro)























**Table of Contents**



- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)

## Repository Structure

Here's an overview of the project's file structure:

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
    - DISCLAIMER: the files **`draw_xxx.py`** are only included in the **`Volunteers-Data`** folder. This script reads images from a specified directory, applies hand landmark detection, and visualizes the results. 

- **`Inference`**: Contains the trained models used in the first approach into the real time LENSEGUA translation.
    - **`ModelXXXP.p`** and **`ModelXXXV.p`**: Pre-trained models for each sign category in each image colection.
    - **`inference_classifier.py`**: Performs real-time hand, face, and body landmark detection and classifies hand gestures based on different combinations of hand, face, and body landmarks. The predictions are made using pre-trained models, and results are displayed on a video feed.

- **`Graphic Interface`**: Contains the assets needed for the user interface to work.
    - **`assets`**: Contains the font used in the interface.
    - **`audio`**: Contains audio files reproduced when audio translation is selected. 
    - **`ModelXXXP.p`** and **`ModelXXXV.p`**: Pre-trained models for each sign category in each image colection.
    - **`gui.py`**: Integrates the real-time video capture, processes it to detect hand signs, and classifies these signs using pre-trained models. It displays real-time translations in text and provides audio playback for each sign. Users can toggle the visibility of detected landmarks (hands, face, pose) with switches on the interface.
    - **`Other files`**: images used in the user interface.
 
## Dependencies

The project depends on multiple Python modules, you can find the libraries and packages you need in **`requirements.txt`**.

| File name | Description                | Folder usage        | Differences| 
| :-------- | :------------------------- | :------- |:-----------------------|
| `collect_imgs.py` | Collects labeled image data.| Static signs ||
| `create_dataset.py` | Creates datasets by extracting hand landmarks. | Static signs ||
| `collect_sequences.py` | Captures frame sequences and automatically extracts holistic landmarks | Dyamic signs ||
| `train_classifier.py` | Trains and evaluates a classifier model for sign language prediction  | Static signs, Dyamic signs |Static signs trains RandomForest, dynamic trains LSTM.|






