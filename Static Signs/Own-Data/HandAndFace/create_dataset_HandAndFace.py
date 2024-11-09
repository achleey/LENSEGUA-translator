import os  # Para manejar directorios y rutas de archivos
import pickle  # Para guardar datos en formato binario
import mediapipe as mp  # Para detección de landmarks en manos y rostros
import cv2  # Para procesamiento de imágenes

# Inicializar modelos de Mediapipe para detección de manos y rostro
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Directorio de datos
DATA_DIR = './HandAndFace'

# Inicializar listas para datos y etiquetas
data = []
labels = []

# Recorrer cada subdirectorio (clase) en DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):  # Verificar que sea un directorio
        for img_path in os.listdir(dir_path):
            combination = []  # Lista para almacenar datos de una imagen
            img = cv2.imread(os.path.join(dir_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB

            # Procesar detección de manos
            resultsH = hands.process(img_rgb)
            hand_data_aux = []
            if resultsH.multi_hand_landmarks:  # Si se detectan manos
                for hand_landmarks in resultsH.multi_hand_landmarks:
                    hand_data = []
                    # Extraer coordenadas x, y de cada landmark de la mano
                    for i in range(len(hand_landmarks.landmark)):
                        xhand = hand_landmarks.landmark[i].x
                        yhand = hand_landmarks.landmark[i].y
                        hand_data.extend([xhand, yhand])
                    hand_data_aux.append(hand_data)

                # Normalizar longitud de datos de manos
                max_length_hand = max(len(seq) for seq in hand_data_aux)
                for seq in hand_data_aux:
                    seq += [0, 0] * (max_length_hand - len(seq))

            # Procesar detección de rostro
            resultsF = face.process(img_rgb)
            face_data_aux = []
            if resultsF.multi_face_landmarks:  # Si se detectan rostros
                for face_landmarks in resultsF.multi_face_landmarks:
                    face_data = []
                    # Extraer coordenadas x, y de cada landmark del rostro
                    for i in range(len(face_landmarks.landmark)):
                        xface = face_landmarks.landmark[i].x
                        yface = face_landmarks.landmark[i].y
                        face_data.extend([xface, yface])
                    face_data_aux.append(face_data)

                # Normalizar longitud de datos de rostro
                max_length_face = max(len(seq) for seq in face_data_aux)
                for seq in face_data_aux:
                    seq += [0, 0] * (max_length_face - len(seq))

            # Unir datos de manos y rostro si ambos están disponibles
            if hand_data_aux and face_data_aux:
                for hand_seq, face_seq in zip(hand_data_aux, face_data_aux):
                    combination = hand_seq + face_seq  # Combinar datos de mano y rostro
                    data.append(combination)
                    labels.append(dir_)  # Agregar etiqueta de la clase

# Guardar datos y etiquetas en un archivo pickle
with open('DataHandAndFaceP.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
