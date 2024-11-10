import os  # Importa el módulo para interactuar con el sistema de archivos
import pickle  # Importa el módulo para serializar y deserializar objetos

import mediapipe as mp  # Importa la librería MediaPipe para la detección de puntos clave (landmarks)
import cv2  # Importa OpenCV para procesar imágenes y video

import numpy as np  # Importa NumPy para trabajar con arrays y operaciones matemáticas

mp_holistic = mp.solutions.holistic  # Accede a la solución holistic de MediaPipe (detectar cuerpo, manos y rostro)
mp_drawing = mp.solutions.drawing_utils  # Utilidades de dibujo de MediaPipe para visualizar los landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # Estilos de dibujo para los landmarks

# Configura el modelo Holistic de MediaPipe
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

DATA_DIR = './Videos'  # Define el directorio de los videos desde donde se cargan las imágenes

data = []  # Lista donde se almacenarán los keypoints extraídos de las imágenes
labels = []  # Lista donde se almacenarán las etiquetas correspondientes a cada secuencia de imágenes

# Función para extraer los puntos clave de las imágenes procesadas por MediaPipe
def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 2)  # 33 landmarks, 2 coords (x, y)
    face = np.array([[res.x, res.y] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 2)  # 468 landmarks, 2 coords (x, y)
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 2)  # 21 landmarks, 2 coords (x, y)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 2)  # 21 landmarks, 2 coords (x, y)
    return np.concatenate([pose, face, lh, rh])  # Devuelve un array concatenado con los puntos clave de cuerpo, rostro y manos

total_samples = 0  # Inicializa el contador para el total de muestras procesadas

# Itera a través de las clases y secuencias almacenadas en el directorio de datos
for class_dir in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_dir)  # Obtiene la ruta completa de la clase
    if os.path.isdir(class_path):  # Verifica si es un directorio de clase
        # Itera a través de las secuencias dentro de cada clase
        for sequence_dir in os.listdir(class_path):
            sequence_path = os.path.join(class_path, sequence_dir)  # Obtiene la ruta completa de la secuencia
            if os.path.isdir(sequence_path):  # Verifica si es un directorio de secuencia

                # Itera a través de los archivos de imágenes en la secuencia, ordenándolos
                for img_file in sorted(os.listdir(sequence_path)):  # Procesa las imágenes en orden para mantener la secuencia
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filtra solo archivos de imagen
                        continue  # Salta archivos que no son imágenes

                    img_path = os.path.join(sequence_path, img_file)  # Obtiene la ruta completa de la imagen
                    img = cv2.imread(img_path)  # Carga la imagen usando OpenCV
                    if img is None:  # Si no se pudo cargar la imagen, muestra un mensaje de advertencia
                        print(f"Warning: no se pudo cargar la imagen {img_path}")
                        continue  # Salta la imagen si no se pudo cargar

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convierte la imagen de BGR a RGB para MediaPipe

                    results = holistic.process(img_rgb)  # Procesa la imagen con el modelo Holistic
                    keypoints = extract_keypoints(results)  # Extrae los puntos clave de los resultados
                    data.append(keypoints)  # Agrega los puntos clave extraídos a la lista de datos
                    labels.append(class_dir)  # Agrega la etiqueta de la clase a la lista de etiquetas

# Guarda los datos y las etiquetas en un archivo pickle
with open('DataHolisticP.pickle', 'wb') as f:  # Abre el archivo para guardar los datos
    pickle.dump({'data': data, 'labels': labels}, f)  # Guarda los datos y las etiquetas en el archivo pickle
