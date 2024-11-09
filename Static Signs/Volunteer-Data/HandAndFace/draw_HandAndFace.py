import os  # Para manejar operaciones relacionadas con el sistema de archivos
import mediapipe as mp  # Biblioteca para detección de características faciales y de manos
import cv2  # Para procesamiento de imágenes con OpenCV
import matplotlib.pyplot as plt  # Para mostrar imágenes

# Inicializar soluciones de MediaPipe para detección de manos y malla facial
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()  # Detectar malla facial

# Utilidades de dibujo de MediaPipe para visualizar los resultados
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicializar la detección de manos con configuración de imagen estática
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directorio que contiene las imágenes de manos y caras
DATA_DIR = './HandAndFace'
specific_dir = '3'  # Subdirectorio específico para procesar imágenes

# Generar la ruta al subdirectorio
dir_path = os.path.join(DATA_DIR, specific_dir)
if os.path.isdir(dir_path):  # Verificar si la ruta es un directorio válido
    # Listar y ordenar archivos de imagen (solo imágenes con nombres numéricos)
    img_files = sorted(
        [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg')) and f.split('.')[0].isdigit()],
        key=lambda x: int(x.split('.')[0]))  # Ordenar las imágenes por nombre numérico

    # Procesar las primeras 20 imágenes
    for img_path in img_files[:20]:
        img = cv2.imread(os.path.join(dir_path, img_path))  # Leer la imagen
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir la imagen a formato RGB

        # Procesar la imagen para obtener características de las manos
        resultsH = hands.process(img_rgb)
        if resultsH.multi_hand_landmarks:  # Si se detectan manos
            for hand_landmarks in resultsH.multi_hand_landmarks:
                # Dibujar los landmarks de las manos en la imagen
                mp_drawing.draw_landmarks(
                    img_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,  # Dibujar las conexiones de los landmarks
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Procesar la imagen para obtener características de la cara
        resultsF = face_mesh.process(img_rgb)
        if resultsF.multi_face_landmarks:  # Si se detectan caras
            for face_landmarks in resultsF.multi_face_landmarks:
                # Dibujar los landmarks de la cara en la imagen
                mp_drawing.draw_landmarks(
                    img_rgb,
                    face_landmarks,
                    mp_face.FACEMESH_CONTOURS  # Dibujar los contornos de la malla facial
                )

        # Mostrar la imagen procesada
        plt.figure()
        plt.imshow(img_rgb)

    plt.show()  # Mostrar todas las imágenes procesadas
