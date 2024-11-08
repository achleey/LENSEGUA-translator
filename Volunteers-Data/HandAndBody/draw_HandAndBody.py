import os  # Para manipulación de archivos y directorios
import mediapipe as mp  # Para el procesamiento de manos y cuerpo
import cv2  # Para leer y procesar imágenes
import matplotlib.pyplot as plt  # Para mostrar las imágenes

# Inicialización de soluciones de MediaPipe para detección de manos y cuerpo
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Utilidades de MediaPipe para dibujar los landmarks
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuración de los objetos para detección de manos y cuerpo
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

# Directorio que contiene las imágenes
DATA_DIR = './HandAndBody'
specific_dir = '0'  # Directorio específico a procesar

# Crear la ruta al directorio específico
dir_path = os.path.join(DATA_DIR, specific_dir)

# Verificar si la ruta es un directorio válido
if os.path.isdir(dir_path):
    # Obtener y ordenar las imágenes del directorio por nombre (numérico)
    img_files = sorted(
        [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg')) and f.split('.')[0].isdigit()],
        key=lambda x: int(x.split('.')[0])
    )

    # Procesar las primeras 5 imágenes del directorio
    for img_path in img_files[:5]:  # Procesar x imágenes
        img = cv2.imread(os.path.join(dir_path, img_path))  # Leer la imagen
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir la imagen a RGB

        # Procesar los landmarks de la mano
        resultsH = hands.process(img_rgb)
        if resultsH.multi_hand_landmarks:  # Si se detectan landmarks de manos
            for hand_landmarks in resultsH.multi_hand_landmarks:
                # Dibujar los landmarks de la mano
                mp_drawing.draw_landmarks(
                    img_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Procesar los landmarks del cuerpo
        resultsP = pose.process(img_rgb)
        if resultsP.pose_landmarks:  # Si se detectan landmarks del cuerpo
            # Dibujar los landmarks del cuerpo
            mp_drawing.draw_landmarks(
                img_rgb,
                resultsP.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Mostrar la imagen con los landmarks dibujados
        plt.figure()
        plt.imshow(img_rgb)

    plt.show()  # Mostrar todas las imágenes procesadas
