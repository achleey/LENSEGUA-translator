import os
import pickle

import matplotlib.pyplot as plt
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

DATA_DIR = './HandAndBody'

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

            resultsP = pose.process(img_rgb)
            pose_data_aux = []
            if resultsP.pose_landmarks:
                pose_data = []
                for pose_landmark in resultsP.pose_landmarks.landmark:
                    xp = pose_landmark.x
                    yp = pose_landmark.y
                    pose_data.extend([xp, yp])
                pose_data_aux.append(pose_data)

                max_length_pose = max(len(seq) for seq in pose_data_aux)
                for seq in pose_data_aux:
                    seq += [0, 0] * (max_length_pose - len(seq))

            if hand_data_aux and pose_data_aux:
                for hand_seq, pose_seq in zip(hand_data_aux, pose_data_aux):
                    combination = hand_seq + pose_seq
                    data.append(combination)
                    labels.append(dir_)

with open('DataHandAndBodyV.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)