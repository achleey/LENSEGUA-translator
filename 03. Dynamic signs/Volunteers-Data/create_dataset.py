import os        # Para manejar rutas de archivos y directorios
import pickle        # Para serializar y guardar los datos procesados

import mediapipe as mp        # Para detección de landmarks (pose, cara, manos) en imágenes
import cv2         # Para procesamiento de imágenes
import numpy as np        # Para manipulación de arrays numéricos

# Inicialización de soluciones de Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuración del modelo Holistic de Mediapipe
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Directorio donde están guardadas las secuencias de imágenes
DATA_DIR = './Sequences'

# Listas para almacenar los datos de puntos clave y sus etiquetas correspondientes
data = []
labels = []

def extract_keypoints(results):
    """Extrae puntos clave (landmarks) de la pose, cara, mano izquierda y mano derecha."""
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 2)
    face = np.array([[res.x, res.y] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 2)
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 2)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 2)
    return np.concatenate([pose, face, lh, rh])

total_samples = 0  # Contador para el total de muestras procesadas

# Iteración por cada clase en el directorio de datos
for class_dir in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_dir)
    if os.path.isdir(class_path):
        
        # Iteración por cada secuencia en la clase
        for sequence_dir in os.listdir(class_path):
            sequence_path = os.path.join(class_path, sequence_dir)
            if os.path.isdir(sequence_path):
                
                # Iteración por cada imagen en la secuencia (procesamiento en orden)
                for img_file in sorted(os.listdir(sequence_path)):
                    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue  # Saltar archivos que no son imágenes
                    
                    # Cargar y procesar la imagen
                    img_path = os.path.join(sequence_path, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: no se pudo cargar la imagen {img_path}")
                        continue

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = holistic.process(img_rgb)  # Procesar la imagen para detectar puntos clave
                    
                    # Extraer y almacenar puntos clave y etiqueta
                    keypoints = extract_keypoints(results)
                    data.append(keypoints)
                    labels.append(class_dir)

# Guardar los datos y etiquetas en un archivo .pickle
with open('DataHolisticV.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
