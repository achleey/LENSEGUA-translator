import os  # Módulo para manipulación de archivos y directorios
import pickle  # Para guardar y cargar datos en archivos binarios

import matplotlib.pyplot as plt  # Para visualizar gráficos (no usado en el código)
import mediapipe as mp  # Librería para procesar landmarks en imágenes
import cv2  # Librería de visión por computadora

# Inicializar el modelo de detección de manos de mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Inicializar el modelo de detección de pose de mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

# Utilidades de dibujo de mediapipe para visualización (no usado en el código)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Ruta donde se almacenan las imágenes para procesamiento
DATA_DIR = './HandAndBody'

# Listas para almacenar los datos y etiquetas
data = []
labels = []

# Recorrer las carpetas de cada clase en el directorio de datos
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):  # Verificar si es un directorio
        for img_path in os.listdir(dir_path):  # Recorrer cada imagen en el directorio
            combination = []  # Lista para almacenar datos combinados de mano y cuerpo
            img = cv2.imread(os.path.join(dir_path, img_path))  # Leer imagen
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB para mediapipe

            # Procesar imagen para detectar landmarks de manos
            resultsH = hands.process(img_rgb)
            hand_data_aux = []
            if resultsH.multi_hand_landmarks:  # Si se detectan manos
                for hand_landmarks in resultsH.multi_hand_landmarks:
                    hand_data = []
                    for i in range(len(hand_landmarks.landmark)):
                        xhand = hand_landmarks.landmark[i].x
                        yhand = hand_landmarks.landmark[i].y
                        hand_data.extend([xhand, yhand])  # Agregar coordenadas (x, y) de cada landmark de mano
                    hand_data_aux.append(hand_data)  # Agregar datos de la mano a la lista auxiliar

                # Rellenar secuencias de manos para que tengan la misma longitud
                max_length_hand = max(len(seq) for seq in hand_data_aux)
                for seq in hand_data_aux:
                    seq += [0, 0] * (max_length_hand - len(seq))

            # Procesar imagen para detectar landmarks de pose (cuerpo)
            resultsP = pose.process(img_rgb)
            pose_data_aux = []
            if resultsP.pose_landmarks:  # Si se detecta la pose
                pose_data = []
                # Extraer coordenadas de hombros y codos
                pose_data.extend([resultsP.pose_landmarks.landmark[11].x, resultsP.pose_landmarks.landmark[11].y])  # Hombro izquierdo
                pose_data.extend([resultsP.pose_landmarks.landmark[12].x, resultsP.pose_landmarks.landmark[12].y])  # Hombro derecho
                pose_data.extend([resultsP.pose_landmarks.landmark[13].x, resultsP.pose_landmarks.landmark[13].y])  # Codo izquierdo
                pose_data.extend([resultsP.pose_landmarks.landmark[14].x, resultsP.pose_landmarks.landmark[14].y])  # Codo derecho

                pose_data_aux.append(pose_data)  # Agregar datos de pose a la lista auxiliar

                # Rellenar secuencias de pose para que tengan la misma longitud
                max_length_pose = max(len(seq) for seq in pose_data_aux)
                for seq in pose_data_aux:
                    seq += [0, 0] * (max_length_pose - len(seq))

            # Combinar datos de manos y pose y guardarlos junto con la etiqueta
            if hand_data_aux and pose_data_aux:
                for hand_seq, pose_seq in zip(hand_data_aux, pose_data_aux):
                    combination = hand_seq + pose_seq
                    data.append(combination)
                    labels.append(dir_)  # Etiqueta de la clase de la imagen

# Guardar los datos y etiquetas en un archivo pickle
with open('DataHandAndBodyP.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
