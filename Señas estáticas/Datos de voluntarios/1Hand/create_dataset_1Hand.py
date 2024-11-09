import os        # Para manejar operaciones del sistema de archivos
import pickle  # Para guardar los datos en formato binario
import mediapipe as mp  # Librería de visión por computadora para detección de manos
import cv2  # Para procesamiento de imágenes

# Inicializar módulos de Mediapipe para detección y dibujo de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configurar detector de manos con modo de imagen estática
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directorio de imágenes de entrada
DATA_DIR = './1Hand'

data = []  # Lista para almacenar las características de las manos
labels = []  # Lista para almacenar etiquetas correspondientes

# Recorrer directorios en DATA_DIR, cada uno representa una clase
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):  # Verifica que sea un directorio

        # Filtrar solo archivos de imagen
        img_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Procesar cada imagen en el directorio
        for img_path in img_files:
            data_aux = []  # Lista auxiliar para almacenar datos de una imagen
            img = cv2.imread(os.path.join(dir_path, img_path))  # Leer imagen
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB para Mediapipe

            results = hands.process(img_rgb)  # Procesar imagen para detectar manos
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_data = []  # Almacenar coordenadas de la mano detectada
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        hand_data.extend([x, y])  # Agregar coordenadas x, y
                    data_aux.append(hand_data)

                # Normalizar longitud de secuencias rellenando con ceros
                max_length = max(len(seq) for seq in data_aux)
                for seq in data_aux:
                    seq += [0, 0] * (max_length - len(seq))
                data.extend(data_aux)  # Añadir datos de la imagen a la lista principal
                labels.extend([dir_] * len(data_aux))  # Añadir etiqueta correspondiente

# Guardar datos y etiquetas en un archivo pickle
with open('Data1HandV.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
