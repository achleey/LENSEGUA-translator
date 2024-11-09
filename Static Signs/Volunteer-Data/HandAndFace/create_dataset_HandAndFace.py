import os  # Para manejar operaciones relacionadas con el sistema de archivos
import pickle  # Para guardar y cargar objetos serializados

import mediapipe as mp  # Biblioteca para procesamiento de imágenes y detección de características
import cv2  # Para procesamiento de imágenes con OpenCV

# Inicializar soluciones de MediaPipe para detección de malla facial y manos
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh()  # Detectar malla facial

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)  # Detectar manos

# Utilidades de dibujo de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Directorio que contiene las imágenes de manos y caras
DATA_DIR = './HandAndFace'

data = []  # Lista para almacenar los datos procesados
labels = []  # Lista para almacenar las etiquetas de cada imagen

# Iterar a través de los directorios de imágenes en DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):  # Si el elemento es un directorio

        # Filtrar solo archivos de imagen en el directorio actual
        img_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for img_path in img_files:
            combination = []  # Lista para combinar características de mano y cara
            img = cv2.imread(os.path.join(dir_path, img_path))  # Leer la imagen
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir la imagen a RGB

            # Procesar la imagen para obtener las características de las manos
            resultsH = hands.process(img_rgb)
            hand_data_aux = []  # Lista para almacenar los datos de las manos
            if resultsH.multi_hand_landmarks:  # Si se detectan manos
                for hand_landmarks in resultsH.multi_hand_landmarks:
                    hand_data = []  # Lista para las coordenadas de una mano
                    for i in range(len(hand_landmarks.landmark)):
                        xhand = hand_landmarks.landmark[i].x  # Coordenada x
                        yhand = hand_landmarks.landmark[i].y  # Coordenada y
                        hand_data.extend([xhand, yhand])  # Agregar las coordenadas a la lista
                    hand_data_aux.append(hand_data)  # Agregar los datos de la mano a la lista general

                # Asegurar que todas las secuencias de coordenadas tengan la misma longitud
                max_length_hand = max(len(seq) for seq in hand_data_aux)
                for seq in hand_data_aux:
                    seq += [0, 0] * (max_length_hand - len(seq))  # Rellenar con ceros si es necesario

            # Procesar la imagen para obtener las características de la cara
            resultsF = face.process(img_rgb)
            face_data_aux = []  # Lista para almacenar los datos de la cara
            if resultsF.multi_face_landmarks:  # Si se detectan caras
                for face_landmarks in resultsF.multi_face_landmarks:
                    face_data = []  # Lista para las coordenadas de la cara
                    for i in range(len(face_landmarks.landmark)):
                        xface = face_landmarks.landmark[i].x  # Coordenada x
                        yface = face_landmarks.landmark[i].y  # Coordenada y
                        face_data.extend([xface, yface])  # Agregar las coordenadas a la lista
                    face_data_aux.append(face_data)  # Agregar los datos de la cara a la lista general

                # Asegurar que todas las secuencias de coordenadas tengan la misma longitud
                max_length_face = max(len(seq) for seq in face_data_aux)
                for seq in face_data_aux:
                    seq += [0, 0] * (max_length_face - len(seq))  # Rellenar con ceros si es necesario

            # Si se encontraron tanto datos de manos como de cara, combinar los datos
            if hand_data_aux and face_data_aux:
                for hand_seq, face_seq in zip(hand_data_aux, face_data_aux):
                    combination = hand_seq + face_seq  # Combinar datos de mano y cara
                    data.append(combination)  # Agregar la combinación a la lista de datos
                    labels.append(dir_)  # Agregar la etiqueta del directorio a la lista de etiquetas

# Guardar los datos y etiquetas en un archivo pickle
with open('DataHandAndFaceV.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
