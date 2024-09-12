import os
import pickle

import mediapipe as mp
import cv2

mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

DATA_DIR = './HandAndFace'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):

        img_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for img_path in img_files:
            combination = []
            img = cv2.imread(os.path.join(dir_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            resultsH = hands.process(img_rgb)
            hand_data_aux = []
            if resultsH.multi_hand_landmarks:
                for hand_landmarks in resultsH.multi_hand_landmarks:
                    hand_data = []
                    for i in range(len(hand_landmarks.landmark)):
                        xhand = hand_landmarks.landmark[i].x
                        yhand = hand_landmarks.landmark[i].y
                        hand_data.extend([xhand, yhand])
                    hand_data_aux.append(hand_data)

                max_length_hand = max(len(seq) for seq in hand_data_aux)
                for seq in hand_data_aux:
                    seq += [0, 0] * (max_length_hand - len(seq))

            resultsF = face.process(img_rgb)
            face_data_aux = []
            if resultsF.multi_face_landmarks:
                for face_landmarks in resultsF.multi_face_landmarks:
                    face_data = []
                    for i in range(len(face_landmarks.landmark)):
                        xface = face_landmarks.landmark[i].x
                        yface = face_landmarks.landmark[i].y
                        face_data.extend([xface, yface])
                    face_data_aux.append(face_data)

                max_length_face = max(len(seq) for seq in face_data_aux)
                for seq in face_data_aux:
                    seq += [0, 0] * (max_length_face - len(seq))

            if hand_data_aux and face_data_aux:
                for hand_seq, face_seq in zip(hand_data_aux, face_data_aux):
                    combination = hand_seq + face_seq
                    data.append(combination)
                    labels.append(dir_)

with open('DataHandAndFaceV.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)