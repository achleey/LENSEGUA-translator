import os  # Para interactuar con el sistema de archivos
import pickle  # Para cargar y guardar objetos en formato pickle

import mediapipe as mp  # Para usar la biblioteca de mediapipe para detección de manos
import cv2  # Para el procesamiento de imágenes

# Inicialización de las herramientas de mediapipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuración del detector de manos
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directorio de datos con imágenes de manos
DATA_DIR = './2Hands'

# Listas para almacenar los datos y etiquetas
data = []
labels = []

# Iterar sobre los subdirectorios dentro de DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)  # Crear la ruta completa del subdirectorio
    if os.path.isdir(dir_path):  # Verificar si es un directorio

        # Obtener archivos de imagen en el subdirectorio
        img_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Procesar cada imagen en el subdirectorio
        for img_path in img_files:
            data_aux = []  # Lista temporal para almacenar los datos de la imagen
            img = cv2.imread(os.path.join(dir_path, img_path))  # Leer la imagen
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir la imagen de BGR a RGB

            results = hands.process(img_rgb)  # Procesar la imagen con mediapipe
            if results.multi_hand_landmarks:  # Si se detectaron manos
                for hand_landmarks in results.multi_hand_landmarks:  # Para cada mano detectada
                    hand_data = []  # Lista para almacenar las coordenadas de los landmarks de la mano
                    for i in range(len(hand_landmarks.landmark)):  # Recorrer cada landmark de la mano
                        x = hand_landmarks.landmark[i].x  # Obtener la coordenada x
                        y = hand_landmarks.landmark[i].y  # Obtener la coordenada y
                        hand_data.extend([x, y])  # Agregar las coordenadas a la lista

                    data_aux.append(hand_data)  # Agregar los datos de la mano a la lista temporal

                # Rellenar las secuencias para que todas tengan la misma longitud
                max_length = max(len(seq) for seq in data_aux)  # Longitud máxima de las secuencias
                for seq in data_aux:
                    seq += [0, 0] * (max_length - len(seq))  # Rellenar con ceros

                # Agregar los datos y las etiquetas (el nombre del directorio) a las listas finales
                data.extend(data_aux)
                labels.extend([dir_] * len(data_aux))

# Guardar los datos y las etiquetas en un archivo pickle
with open('Data2HandsV.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
