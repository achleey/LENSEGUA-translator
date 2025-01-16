
![Logo](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F75qpoqrku7bog2weyyno.jpeg)

# LENSEGUA translator 🇬🇹

This project aims to develop a comprehensive desktop application for translating Guatemalan Sign Language (LENSEGUA) using computer vision and machine learning. The initial version sets up the framework for real-time video processing and sign language recognition, utilizing computer vision techniques to capture and interpret signs. The application integrates machine learning models to translate detected signs into text and audio, providing an interactive and efficient translation tool. The ultimate goal is to create a robust system for translating Guatemalan Sign Language in real-time, facilitating communication and accessibility.

## Table of contents

- [Features](#features)
- [Structure](#structure)
- [Demo](#demo)
- [System requirements](#system-requirements)
- [Acknowledgements](#acknowledgements)
- [Support](#support)
- [License](#license)
  
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
    | `create_dataset.py` | Creates datasets by extracting hand landmarks from images. | Static signs | -|
    | `collect_sequences.py` | Captures labeled frame sequences and automatically extracts holistic landmarks. | Dyamic signs |-|
    | `train_classifier.py` | Trains and evaluates a classifier model for sign language prediction.  | Static signs, Dyamic signs |Static signs trains a RandomForest model, Dynamic trains a LSTM model.|
    | `inference_classifier.py` | Detects and predicts sign language in real time.  | Static signs, Dyamic signs |Static signs does simple hand detection and letter classification using a pickle-loaded model, Dynamics uses holistic detection, and TensorFlow for sequence-based predictions.|

**02\. Static signs:** Consists of 2 subfolders:

- **Own-Data**: It consists of 4 subfolders: **`1Hand`**, **`2Hands`**, **`HandAndFace`** and **`HandAndBody`**. It contains code for collecting images, extracting features using MediaPipe solutions based on the body parts used to gesture signs, and training a classifier model using self-portrait images.

- **Volunteers-Data**: It consists of 4 subfolders: **`1Hand`**, **`2Hands`**, **`HandAndFace`** and **`HandAndBody`**. It contains code for extracting features from existing volunteer images, plotting landmarks based on the body parts used to gesture signs, and training a classifier model. 

    | File name | Description                | Folder usage        | Differences| 
    | :-------- | :------------------------- | :----------- |:-----------------------|
    | `collect_imgs_XXX.py` | Collects labeled image data.| Own-Data (all subfolders) |For 1Hand it collects 16 different sets of labeled image data, for 2Hands it collects 3 sets, for HandAndBody it collects 1 set and for HandAndFace it collects 4 sets.| 
    | `create_dataset_XXX.py` | Creates datasets by extracting specific landmarks from images. | Own-Data, Volunteer-Data (all subfolders for both) | 1Hand and 2Hands use `Hands`, HandAndBody uses `Hands` and `Pose`, and HandAndFace uses `Hands`and `FaceMesh`. It returns the dataset in a `XHandXXX.pickle` format.|
    | `draw_XXX.py` | Reads images from a specified directory, applies landmark detection, and plots said landmarks. | Volunteer-Data (all subfolders) | 1Hand and 2Hands plots hand landmarks, HandAndBody plots hand and pose landmarks, and HandAndFace plots hand and face landmarks.|
    | `train_classifier.py` | Trains and evaluates a **RandomForest** classifier model for sign language prediction. | Own-Data, Volunteer-Data (all subfolders for both) | The difference consists of the `.pickle` dataset loaded for training. It returns the model in a `ModelXXX.p` format.|

**03\. Dynamic signs:** Consists of 2 subfolders: 

- **Own-Data**: Contains code for collecting video sequences as frames, extracting features using MediaPipe Holistic, training a **LSTM** classifier model and converting said model to a TensorFlow Lite format.

- **Volunteers-Data**: Contains code for converting existing videos into frame sequences, extracting features using MediaPipe Holistic, training a **LSTM** classifier model and converting said model to a TensorFlow Lite format.

    | File name | Description                | Folder usage        | Differences| 
    | :-------- | :------------------------- | :------- |:-----------------------|
    | `collect_sequences.py` | Captures labeled frame sequences.| Own-Data |-| 
    | `create_sequences.py` | Creates frame sequences from existing videos. | Volunteers-Data |-|
    | `create_dataset.py` | Creates datasets by extracting holistic landmarks from frame sequences. | Own-Data, Volunteer-Data|It creates a `DataHolisticX.pickle`|
    | `train_classifier.py` | Trains and evaluates a LSTM classifier model for sign language prediction.| Own-Data, Volunteer-Data| The difference consists of the `.pickle` dataset loaded for training. It returns a `ModelHolisticX` folder in a SavedModel format.|
    | `lite_model.py` | Converts SavedModel folders in to a TensorFlow Lite model.| Own-Data, Volunteer-Data| The difference consists of the `ModelHolisticX` folder loaded for converting. It returns an `actionX.tflite` model.|

**04\. Real time inference**: Consists of 2 subfolders:

- **Static inference:** Contains code for real-time detection and prediction of static signs, along with the pre-trained **RandomForest** models used for predictions.

- **Dynamic inference:** Contains code for real-time detection and prediction of dynamic signs, along with the pre-trained **LSTM** TensorFlow Lite models used for predictions.

    | File name | Description                | Folder usage        | Differences| 
    | :-------- | :------------------------- | :------- |:-----------------------|
    | `inference_classifier.py` | Detects and predicts sign language in real time.| Static inference, Dynamic inference | The difference consists on the models loaded for prediction and the type of signs (static or dynamic) translated.|

**05\. Graphic interface:** Contains the code to run a graphic interface that enables sign language translation to text and audio, as well as optional visualization of landmarks in real time. It also includes trained models, images, fonts, and audio files used in the interface.

  | File name | Description                |
  | :-------- | :------------------------- |
  | `gui_static.py` |  Runs the graphic interface performing only static sign language translation.| 
  | `gui_complete.py` |  Runs the graphic interface performing both static and dynamic sign language translation.|

**06\. App development:** This folder contains the code to package a Python application for macOS in Alias mode. Note that this prototype is not intended for distribution. The contents of this folder should be executed alongside the graphic interface resources. 

  | File name | Description                |
  | :-------- | :------------------------- |
  | `setup.py` |  Defines the application, its dependencies, resources, and configuration options to create a standalone macOS application bundle.| 
  | `Info.plist` | Defines the app's metadata, including its name, version, bundle identifier, system requirements, and permissions like camera and microphone access.|
  | `app.png` | Icon image in PNG format.|
  | `app.icns` | Icon image in icns format, compatible with macOS.|

**Additional files:**

- **`requirements.txt`**: This file contains a list of all the dependencies required to run the project.

## Demo

- Static Signs
  
[![Static Signs](https://img.youtube.com/vi/VKFGZf9meWA/0.jpg)](https://www.youtube.com/watch?v=VKFGZf9meWA)

- Dynamic Signs
  
[![Dynamic Signs](https://img.youtube.com/vi/kzNIpRGYnkI/0.jpg)](https://www.youtube.com/watch?v=kzNIpRGYnkI)

- Prototype: Desktop App
  
[![Desktop App](https://img.youtube.com/vi/vn-zZjlV7BI/0.jpg)](https://www.youtube.com/watch?v=vn-zZjlV7BI)

## System requirements

This project was developed in a Macbook Pro (Apple Silicon M1 Pro Chip) with macOS Sonoma 14.5. The following software is needed 🛠️:

- Python 3.10
- PyCharm 2024.1 (Professional Edition), or another IDE (as long as the migration process is done correctly).

## Acknowledgements

This section includes resources and references that contributed to the development of the project:

- 🎖️ **Static signs** :
    
    - [Sign Language Detector - GitHub](https://github.com/computervisioneng/sign-language-detector-python)
    - [Sign Language Detector - YouTube](https://www.youtube.com/watch?v=MJCSjXepaAM&t=1072s)

- 🎖️ **Dynamic signs**:

    - [Action Recognition Sign Language Detection - GitHub](https://github.com/nicknochnack/ActionDetectionforSignLanguage)
    - [Action Recognition Sign Language Detection - YouTube](https://www.youtube.com/watch?v=doDUihpj6ro)

## Support

👩🏻‍💻 Author: Ashley Morales

If you have any questions, please contact ashleymoralesaldana@gmail.com.
 
## License

[MIT](https://choosealicense.com/licenses/mit/)






