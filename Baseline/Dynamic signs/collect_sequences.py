import cv2  # Importar la biblioteca OpenCV para procesamiento de imágenes y video
import numpy as np  # Importar la biblioteca NumPy para manipulación de arrays
import os  # Importar la biblioteca para manejar rutas y archivos en el sistema operativo
import mediapipe as mp  # Importar Mediapipe, una librería de Google para la detección de puntos clave en imágenes

# Configuración de Mediapipe
mp_holistic = mp.solutions.holistic  # Inicializar el modelo Holistic de Mediapipe, que detecta cuerpo, cara, manos, etc.
mp_drawing = mp.solutions.drawing_utils  # Utilizar las utilidades de dibujo de Mediapipe para dibujar los puntos clave

# Función para detección con Mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir la imagen de BGR (OpenCV) a RGB (Mediapipe)
    image.flags.writeable = False  # Deshabilitar la escritura de la imagen, mejora la eficiencia durante el procesamiento
    results = model.process(image)  # Ejecutar el modelo para obtener los resultados de detección de puntos clave
    image.flags.writeable = True  # Habilitar la escritura nuevamente después de la detección
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Volver a convertir la imagen de RGB a BGR para mostrarla correctamente
    return image, results  # Devolver la imagen procesada y los resultados de la detección

# Función para dibujar landmarks estilizados
def draw_styled_landmarks(image, results):
    # Dibujar los puntos clave de la cara, utilizando las conexiones del mesh de la cara
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS)
    # Dibujar los puntos clave del cuerpo, utilizando las conexiones del pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # Dibujar los puntos clave de la mano izquierda
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Dibujar los puntos clave de la mano derecha
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Función para extraer puntos clave
def extract_keypoints(results):
    # Extraer las coordenadas x, y, z, y la visibilidad de los puntos clave del cuerpo (pose)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    # Extraer las coordenadas x, y, z de los puntos clave de la cara
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    # Extraer las coordenadas x, y, z de los puntos clave de la mano izquierda
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    # Extraer las coordenadas x, y, z de los puntos clave de la mano derecha
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    # Concatenar todos los puntos clave extraídos (cuerpo, cara, mano izquierda, mano derecha) en un solo array
    return np.concatenate([pose, face, lh, rh])

# Configuración de rutas y datos
DATA_PATH = os.path.join('MP_Data')  # Definir la ruta donde se guardarán los datos procesados
actions = np.array(['F', 'J', 'S'])  # Definir las acciones que se quieren detectar y guardar
no_sequences = 168  # Número de secuencias (videos) por cada acción
sequence_length = 30  # Longitud de cada secuencia (video) en frames
start_folder = 0  # Carpeta de inicio para almacenar las secuencias de datos

# Crear carpetas para almacenar datos si no existen
for action in actions:
    action_path = os.path.join(DATA_PATH, action)  # Crear la ruta para la acción actual
    if not os.path.exists(action_path):
        os.makedirs(action_path)  # Crear la carpeta si no existe
    dirmax = max([int(folder) for folder in os.listdir(action_path) if folder.isdigit()], default=0)  # Obtener el número de la última carpeta creada

    for sequence in range(1, no_sequences + 1):
        sequence_path = os.path.join(action_path, str(dirmax + sequence))  # Crear la ruta para cada secuencia de datos
        if not os.path.exists(sequence_path):
            os.makedirs(sequence_path)  # Crear la carpeta para la secuencia si no existe

# Captura de video y recolección de datos
cap = cv2.VideoCapture(1)  # Inicializar la captura de video (dispositivo 1, generalmente la cámara secundaria)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        # Espera a que se presione 's' para comenzar la captura de datos de cada acción
        print(f"Presione 's' para comenzar la captura de datos para la acción '{action}' o 'q' para salir.")
        while True:
            ret, frame = cap.read()  # Leer un frame del video
            image, results = mediapipe_detection(frame, holistic)  # Realizar la detección con Mediapipe
            draw_styled_landmarks(image, results)  # Dibujar los puntos clave estilizados sobre la imagen

            # Mostrar un mensaje en pantalla con las instrucciones
            cv2.putText(image, f'Presione "s" para {action} o "q" para salir', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)  # Mostrar la imagen procesada en una ventana

            key = cv2.waitKey(10) & 0xFF  # Esperar una tecla por 10ms
            if key == ord('s'):
                break  # Iniciar la captura de datos para la acción actual cuando se presione 's'
            elif key == ord('q'):
                cap.release()  # Liberar el dispositivo de captura si se presiona 'q'
                cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV
                exit()  # Salir del programa

        for sequence in range(start_folder, start_folder + no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()  # Leer un nuevo frame del video
                image, results = mediapipe_detection(frame, holistic)  # Realizar la detección de puntos clave
                draw_styled_landmarks(image, results)  # Dibujar los puntos clave estilizados sobre la imagen

                if frame_num == 0:
                    # Mostrar mensajes indicando el inicio de la colección de datos para una secuencia
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)  # Mostrar la imagen con los mensajes
                    cv2.waitKey(500)  # Esperar medio segundo para visualizar los mensajes antes de comenzar la recolección de datos
                else:
                    # Mostrar solo el mensaje de recolección de frames para la secuencia
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)  # Mostrar la imagen procesada

                # Verificar si la subcarpeta para la secuencia existe, si no, crearla
                npy_path = os.path.join(DATA_PATH, action, str(sequence))
                if not os.path.exists(npy_path):
                    os.makedirs(npy_path)  # Crear la carpeta si no existe

                # Guardar los puntos clave extraídos en formato numpy (.npy)
                np.save(os.path.join(npy_path, str(frame_num)), extract_keypoints(results))

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break  # Salir del ciclo si se presiona 'q'

cap.release()  # Liber
cv2.destroyAllWindows()
