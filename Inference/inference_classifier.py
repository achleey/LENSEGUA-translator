import pickle

import cv2
import mediapipe as mp
import numpy as np

#Modelo para 1 mano
model_dict1 = pickle.load(open('./Model1HandP.p', 'rb')) #Cambiar modelos segun el conjunto de datos que se estén utilizando. Ej: Model1HandV.p
model1 = model_dict1['model']

#Modelo para 2 manos
model_dict2 = pickle.load(open('./Model2HandsP.p', 'rb'))
model2 = model_dict2['model']

#Modelo para mano y cara
model_dict3 = pickle.load(open('./ModelHandAndFaceP.p', 'rb'))
model3 = model_dict3['model']

#Modelo para mano y cuerpo
model_dict4 = pickle.load(open('./ModelHandAndBodyP.p', 'rb'))
model4 = model_dict4['model']

labels_dict = {0: 'A', 1: 'B', 2:'C', 3:'D', 4: 'E', 5:'K', 6:'L', 7:'M', 8: 'N', 9:'O', 10: 'P', 11: 'R', 12: 'U', 13:'V', 14: 'W', 15: 'Y'}
labels_dict2 = {0: 'Ñ', 1: 'Q', 2:'X'}
labels_dict3 = {0: 'G', 1: 'H', 2:'I', 3: 'T'}
labels_dict4 = {0: 'Z'}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(1)

def is_hand_near_face(hand_landmarks, face_landmarks, threshold=0.03):
    for hand_point in hand_landmarks:
        for face_point in face_landmarks:
            distance = np.sqrt((hand_point.x - face_point.x)**2 + (hand_point.y - face_point.y)**2)
            if distance < threshold:
                return True
    return False


def is_hand_horizontal(hand_landmarks, angle_threshold=45):
    if not hand_landmarks:
        return False

    landmarks = hand_landmarks.landmark
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    # Calcular el ángulo de la mano respecto al eje horizontal
    dx = index_finger_tip.x - pinky_tip.x
    dy = index_finger_tip.y - pinky_tip.y
    angle = np.arctan2(dy, dx) * 180 / np.pi

    return np.abs(angle) < angle_threshold or np.abs(angle - 180) < angle_threshold

def is_elbow_visible(pose_landmarks, hand_landmarks, pose_threshold=0.5, hand_threshold=45):
    if not pose_landmarks or not hand_landmarks:
        return False

    landmarks = pose_landmarks.landmark
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

    left_elbow_visible = left_elbow.visibility > pose_threshold
    right_elbow_visible = right_elbow.visibility > pose_threshold

    hand_horizontal = is_hand_horizontal(hand_landmarks, angle_threshold=hand_threshold)

    return (left_elbow_visible or right_elbow_visible) and hand_horizontal

while True:

    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resultsH = hands.process(frame_rgb)
    resultsF = face.process(frame_rgb)
    resultsP = pose.process(frame_rgb)

    num_hands = 0
    face_detected = False
    face_data_aux = []
    hand_data_aux = []
    pose_data_aux = []

    if resultsH.multi_hand_landmarks:
        num_hands = len(resultsH.multi_handedness)
        for hand_landmarks in resultsH.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            hand_data_aux = []
            for landmark in hand_landmarks.landmark:
                xh = landmark.x
                yh = landmark.y
                hand_data_aux.append(xh)
                hand_data_aux.append(yh)

    if resultsF.multi_face_landmarks:
        face_detected = True
        for face_landmarks in resultsF.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_CONTOURS,
                mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )

            for landmark in face_landmarks.landmark:
                xf = landmark.x
                yf = landmark.y
                face_data_aux.append(xf)
                face_data_aux.append(yf)

    if resultsP.pose_landmarks:
        pose_data = []
        mp_drawing.draw_landmarks(
            frame,
            resultsP.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        pose_data_aux = []
        landmarks = resultsP.pose_landmarks.landmark
        pose_data_aux.append(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x)
        pose_data_aux.append(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y)
        pose_data_aux.append(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x)
        pose_data_aux.append(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y)
        pose_data_aux.append(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x)
        pose_data_aux.append(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
        pose_data_aux.append(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
        pose_data_aux.append(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)

    predicted_character = 'No se detecta una combinación válida'

    #if num_hands == 1:
            #prediction = model1.predict([np.asarray(hand_data_aux)])
            #predicted_character = labels_dict[int(prediction[0])]

    #elif num_hands == 2:
            #prediction = model2.predict([np.asarray(hand_data_aux)])
            #predicted_character = labels_dict2[int(prediction[0])]

    if num_hands >= 1 and face_detected:
        if is_hand_near_face(resultsH.multi_hand_landmarks[0].landmark, resultsF.multi_face_landmarks[0].landmark):
            combination = hand_data_aux + face_data_aux
            prediction = model3.predict([np.asarray(combination)])
            predicted_character = labels_dict3[int(prediction[0])]
        else:
            if num_hands == 1:
                prediction = model1.predict([np.asarray(hand_data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
            else:
                prediction = model2.predict([np.asarray(hand_data_aux)])
                predicted_character = labels_dict2[int(prediction[0])]
    if num_hands == 1 and is_elbow_visible(resultsP.pose_landmarks, resultsH.multi_hand_landmarks[0]):
        combinationByH = hand_data_aux + pose_data_aux
        prediction = model4.predict([np.asarray(combinationByH)])
        predicted_character = labels_dict4[int(prediction[0])]

    if hand_data_aux:
        x1 = int(min(np.asarray(hand_data_aux)[::2]) * W)
        y1 = int(min(np.asarray(hand_data_aux)[1::2]) * H)
        x2 = int(max(np.asarray(hand_data_aux)[::2]) * W)
        y2 = int(max(np.asarray(hand_data_aux)[1::2]) * H)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 5)
        cv2.putText(frame, predicted_character, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 5,
                    (0, 0, 0), 5, cv2.LINE_AA)
    else:
        cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.13,
                    (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
