import os                    # Para operaciones de archivos y directorios
import pickle                # Para guardar y cargar datos en archivos binarios
import mediapipe as mp       # Libería de procesamiento de imágenes y detección de manos
import cv2                   # Para lectura y manipulación de imágenes

# Configuración de mediapipe para detección de manos y dibujo de landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicialización del modelo de detección de manos en modo de imagen estática
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directorio donde están almacenadas las imágenes de manos
DATA_DIR = './1Hand'

# Listas para almacenar los datos de landmarks y las etiquetas de cada imagen
data = []
labels = []

# Procesar cada directorio de clase dentro de DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):        # Asegura que sea un directorio
        # Procesar cada imagen dentro del directorio de clase
        for img_path in os.listdir(dir_path):
            data_aux = []        # Almacena los datos temporales de la mano en la imagen actual
            img = cv2.imread(os.path.join(dir_path, img_path))        # Cargar imagen
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # Convertir imagen a RGB

            results = hands.process(img_rgb)        # Detección de manos en la imagen
            if results.multi_hand_landmarks:        # Si se detecta al menos una mano
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_data = []
                    # Extraer coordenadas x e y de cada landmark
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        hand_data.extend([x, y])        # Agregar coordenadas a hand_data
                    data_aux.append(hand_data)        # Agregar datos de la mano a data_Aux

                # Ajuste de longitud de secuencias para uniformidad en los datos
                max_length = max(len(seq) for seq in data_aux)    # Longitud máxima de las secuencias
                for seq in data_aux:
                    seq += [0, 0] * (max_length - len(seq))    # Rellenar con ceros hasta la longitud máxima
                data.extend(data_aux)        # Añadir secuencias a los datos principales
                labels.extend([dir_] * len(data_aux))    # Añadir etiquetas correspondientes

# Guardar los datos en un archivo pickle
with open('Data1HandP.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
