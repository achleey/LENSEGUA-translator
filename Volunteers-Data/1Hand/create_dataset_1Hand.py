import os
import pickle

import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './1Hand'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):

        img_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for img_path in img_files:
            data_aux = []
            img = cv2.imread(os.path.join(dir_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_data = []
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        hand_data.extend([x, y])
                    data_aux.append(hand_data)

                max_length = max(len(seq) for seq in data_aux)
                for seq in data_aux:
                    seq += [0, 0] * (max_length - len(seq))
                data.extend(data_aux)
                labels.extend([dir_] * len(data_aux))

# Guardar los datos en un archivo pickle
with open('Data1HandV.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)