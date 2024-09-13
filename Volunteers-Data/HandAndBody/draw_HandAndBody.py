import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './HandAndBody'
specific_dir = '0'

dir_path = os.path.join(DATA_DIR, specific_dir)
if os.path.isdir(dir_path):
    img_files = sorted(
        [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg')) and f.split('.')[0].isdigit()],
        key=lambda x: int(x.split('.')[0]))

    for img_path in img_files[:5]:  # Procesar x imágenes
        img = cv2.imread(os.path.join(dir_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        resultsH = hands.process(img_rgb)
        if resultsH.multi_hand_landmarks:
            for hand_landmarks in resultsH.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        resultsP = pose.process(img_rgb)
        if resultsP.pose_landmarks:
            mp_drawing.draw_landmarks(
                img_rgb,
                resultsP.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style()
            )

        plt.figure()
        plt.imshow(img_rgb)

    plt.show()