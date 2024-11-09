import os        # Para manejo de directorios y archivos
import pickle        # Para guardar y cargar datos en formato binario
import mediapipe as mp        # Librería para procesamiento de landmarks
import cv2        # Para manejo de imágenes y captura de vídeo

# Inicializar módulos de MediaPipe para detección de manos y estilos de dibujo
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configurar el detector de manos en modo estático con un mínimo de confianza de detección
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directorio de imágenes para procesamiento
DATA_DIR = './2Hands'

# Listas para almacenar datos y etiquetas
data = []
labels = []

# Recorrer las carpetas de clases en el directorio de datos
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):
        for img_path in os.listdir(dir_path):
            data_aux = []
            img = cv2.imread(os.path.join(dir_path, img_path))        # Cargar imagen
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # Convertir a RGB

            # Procesar imagen para detectar landmarks de manos
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                # Recorrer cada mano detectada en la imagen
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_data = []
                    for i in range(len(hand_landmarks.landmark)):        # Extraer coordenadas x e y de cada landmark
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        hand_data.extend([x, y])        # Agregar coordenadas a la lista 
                    data_aux.append(hand_data)        # Añadir datos de mano a la lista auxiliar

                # Normalizar la longitud de las secuencias de datos agregando ceros
                max_length = max(len(seq) for seq in data_aux)
                for seq in data_aux:
                    seq += [0, 0] * (max_length - len(seq))
                data.extend(data_aux)        # Agregar datos normalizados a la lista principal
                labels.extend([dir_] * len(data_aux))        # Etiquetar cada dato con el nombre de la clase

# Guardar los datos en un archivo pickle
with open('Data2HandsP.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
