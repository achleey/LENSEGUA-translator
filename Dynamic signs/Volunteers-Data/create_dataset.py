import os
import pickle

import mediapipe as mp
import cv2

import numpy as np

mp_holistic = mp.solutions.holistic

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

DATA_DIR = './Sequences'

data = []
labels = []

def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 2)  # 33 landmarks, 2 coords (x, y)
    face = np.array([[res.x, res.y] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 2)  # 468 landmarks, 2 coords (x, y)
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 2)  # 21 landmarks, 2 coords (x, y)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 2)  # 21 landmarks, 2 coords (x, y)
    return np.concatenate([pose, face, lh, rh])

total_samples = 0  # Contador para total de muestras procesadas

for class_dir in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_dir)
    if os.path.isdir(class_path):
        for sequence_dir in os.listdir(class_path):
            sequence_path = os.path.join(class_path, sequence_dir)
            if os.path.isdir(sequence_path):

                for img_file in sorted(os.listdir(sequence_path)): #En este caso es necesario procesar en orden los datos por que se está trabajando con secuencias.
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue  # Saltar archivos que no son imágenes

                    img_path = os.path.join(sequence_path, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: no se pudo cargar la imagen {img_path}")
                        continue

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    results = holistic.process(img_rgb)
                    keypoints = extract_keypoints(results)
                    data.append(keypoints)
                    labels.append(class_dir)

with open('DataHolisticV.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
