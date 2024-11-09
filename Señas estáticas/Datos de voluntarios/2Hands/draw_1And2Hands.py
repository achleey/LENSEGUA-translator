import os  # Para interactuar con el sistema de archivos
import mediapipe as mp  # Para usar la biblioteca de mediapipe para detección de manos
import cv2  # Para procesamiento de imágenes
import matplotlib.pyplot as plt  # Para visualizar las imágenes

# Inicialización de las herramientas de mediapipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuración del detector de manos
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directorio de datos con imágenes de manos
DATA_DIR = './2Hands'
specific_dir = '2'  # Subdirectorio específico a procesar

# Crear la ruta completa del subdirectorio
dir_path = os.path.join(DATA_DIR, specific_dir)

# Verificar si el subdirectorio existe
if os.path.isdir(dir_path):
    # Obtener archivos de imagen ordenados alfabéticamente que tengan extensiones de imagen válidas
    img_files = sorted(
        [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg')) and f.split('.')[0].isdigit()],
        key=lambda x: int(x.split('.')[0]))  # Ordenar por el número en el nombre del archivo

    # Procesar las primeras 10 imágenes
    for img_path in img_files[:10]:
        # Leer la imagen y convertirla de BGR a RGB
        img = cv2.imread(os.path.join(dir_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con mediapipe
        results = hands.process(img_rgb)
        
        # Si se detectaron landmarks de las manos
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar los landmarks y las conexiones de la mano sobre la imagen
                mp_drawing.draw_landmarks(
                    img_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Mostrar la imagen procesada
        plt.figure()
        plt.imshow(img_rgb)

    # Mostrar todas las imágenes procesadas
    plt.show()
