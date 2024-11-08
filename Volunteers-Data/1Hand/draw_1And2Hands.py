import os  # Para manejar operaciones del sistema de archivos
import mediapipe as mp  # Librería de visión por computadora para la detección de manos
import cv2  # Para procesamiento de imágenes
import matplotlib.pyplot as plt  # Para visualización de imágenes

# Inicialización de mediapipe para la detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuración de la detección de manos
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directorio y subdirectorio específicos de las imágenes
DATA_DIR = './1Hand'
specific_dir = '5'

dir_path = os.path.join(DATA_DIR, specific_dir)

# Verificar si el subdirectorio existe
if os.path.isdir(dir_path):
    # Obtener archivos de imagen, ordenados por el número que aparece en el nombre del archivo
    img_files = sorted(
        [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg')) and f.split('.')[0].isdigit()],
        key=lambda x: int(x.split('.')[0])
    )

    # Procesar las primeras 10 imágenes del directorio
    for img_path in img_files[:10]:
        img = cv2.imread(os.path.join(dir_path, img_path))  # Leer la imagen
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir la imagen a formato RGB

        # Procesar la imagen para detectar las manos
        results = hands.process(img_rgb)

        # Si se detectan manos, dibujar los landmarks y las conexiones
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Mostrar la imagen con los landmarks dibujados
        plt.figure()
        plt.imshow(img_rgb)

    plt.show()  # Mostrar todas las imágenes procesadas
