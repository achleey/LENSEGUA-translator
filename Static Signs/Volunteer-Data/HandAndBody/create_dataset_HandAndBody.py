import os  # Para manipulación de archivos y directorios
import pickle  # Para guardar y cargar objetos de Python
import matplotlib.pyplot as plt  # Para visualización de gráficos
import mediapipe as mp  # Para el procesamiento de manos y cuerpo
import cv2  # Para leer y procesar imágenes

# Inicialización de soluciones de MediaPipe para detección de manos y cuerpo
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

mp_drawing = mp.solutions.drawing_utils  # Utilidades para dibujar los landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # Estilos para dibujar los landmarks

DATA_DIR = './HandAndBody'  # Directorio donde están las imágenes de manos y cuerpo

data = []  # Lista para almacenar los datos procesados
labels = []  # Lista para almacenar las etiquetas (directorios de las imágenes)

# Iterar sobre los subdirectorios dentro del directorio de datos
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)  # Obtener la ruta completa del subdirectorio
    if os.path.isdir(dir_path):  # Verificar si es un directorio

        # Obtener todos los archivos de imagen en el directorio
        img_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Procesar cada imagen en el directorio
        for img_path in img_files:
            combination = []  # Lista para almacenar la combinación de datos de mano y cuerpo
            img = cv2.imread(os.path.join(dir_path, img_path))  # Leer la imagen
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir la imagen a RGB

            # Procesar los landmarks de la mano
            resultsH = hands.process(img_rgb)
            hand_data_aux = []  # Lista temporal para los datos de las manos
            if resultsH.multi_hand_landmarks:  # Si se detectan landmarks de manos
                for hand_landmarks in resultsH.multi_hand_landmarks:
                    hand_data = []  # Lista para los datos de una mano
                    for i in range(len(hand_landmarks.landmark)):  # Iterar sobre los landmarks de la mano
                        xhand = hand_landmarks.landmark[i].x
                        yhand = hand_landmarks.landmark[i].y
                        hand_data.extend([xhand, yhand])  # Agregar las coordenadas (x, y) de cada landmark
                    hand_data_aux.append(hand_data)

                # Asegurar que todos los datos de manos tengan la misma longitud
                max_length_hand = max(len(seq) for seq in hand_data_aux)
                for seq in hand_data_aux:
                    seq += [0, 0] * (max_length_hand - len(seq))  # Rellenar con ceros

            # Procesar los landmarks del cuerpo
            resultsP = pose.process(img_rgb)
            pose_data_aux = []  # Lista temporal para los datos del cuerpo
            if resultsP.pose_landmarks:  # Si se detectan landmarks del cuerpo
                pose_data = []  # Lista para los datos del cuerpo
                # Extraer los landmarks relevantes (hombros y codos)
                pose_data.extend([resultsP.pose_landmarks.landmark[11].x, resultsP.pose_landmarks.landmark[11].y])  # Hombro izquierdo
                pose_data.extend([resultsP.pose_landmarks.landmark[12].x, resultsP.pose_landmarks.landmark[12].y])  # Hombro derecho
                pose_data.extend([resultsP.pose_landmarks.landmark[13].x, resultsP.pose_landmarks.landmark[13].y])  # Codo izquierdo
                pose_data.extend([resultsP.pose_landmarks.landmark[14].x, resultsP.pose_landmarks.landmark[14].y])  # Codo derecho

                pose_data_aux.append(pose_data)

                # Asegurar que todos los datos del cuerpo tengan la misma longitud
                max_length_pose = max(len(seq) for seq in pose_data_aux)
                for seq in pose_data_aux:
                    seq += [0, 0] * (max_length_pose - len(seq))  # Rellenar con ceros

            # Si hay datos tanto de manos como de cuerpo, combinar y agregar a las listas
            if hand_data_aux and pose_data_aux:
                for hand_seq, pose_seq in zip(hand_data_aux, pose_data_aux):
                    combination = hand_seq + pose_seq  # Combinar los datos de la mano y el cuerpo
                    data.append(combination)  # Agregar los datos combinados
                    labels.append(dir_)  # Agregar la etiqueta (nombre del directorio)

# Guardar los datos y etiquetas procesados en un archivo pickle
with open('DataHandAndBodyV.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
