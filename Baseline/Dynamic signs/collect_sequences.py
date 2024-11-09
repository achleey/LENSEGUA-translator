import cv2
import numpy as np
import os
import mediapipe as mp

# Configuración de Mediapipe
mp_holistic = mp.solutions.holistic  # Modelo Holistic
mp_drawing = mp.solutions.drawing_utils  # Utilidades de dibujo


# Función para detección con Mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
    image.flags.writeable = False  # La imagen no es editable
    results = model.process(image)  # Predicción
    image.flags.writeable = True  # La imagen vuelve a ser editable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convertir de RGB a BGR
    return image, results


# Función para dibujar landmarks estilizados
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


# Función para extraer puntos clave
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


# Configuración de rutas y datos
DATA_PATH = os.path.join('MP_Data')  # Ruta para exportar datos como arrays de numpy
actions = np.array(['F', 'J', 'S'])  # Acciones que intentamos detectar
no_sequences = 168  # Número de videos por acción
sequence_length = 30  # Longitud de cada video en frames
start_folder = 0  # Carpeta de inicio para los datos

# Crear carpetas para almacenar datos si no existen
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        os.makedirs(action_path)
    dirmax = max([int(folder) for folder in os.listdir(action_path) if folder.isdigit()], default=0)

    for sequence in range(1, no_sequences + 1):
        sequence_path = os.path.join(action_path, str(dirmax + sequence))
        if not os.path.exists(sequence_path):
            os.makedirs(sequence_path)

# Captura de video y recolección de datos
cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        # Espera a que se presione 's' para comenzar la captura de datos de cada acción
        print(f"Presione 's' para comenzar la captura de datos para la acción '{action}' o 'q' para salir.")
        while True:
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Mostrar mensaje en pantalla
            cv2.putText(image, f'Presione "s" para {action} o "q" para salir', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                break  # Iniciar la captura de datos para la acción actual
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        for sequence in range(start_folder, start_folder + no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else:
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                # Verificar y crear subcarpeta para cada frame antes de guardar
                npy_path = os.path.join(DATA_PATH, action, str(sequence))
                if not os.path.exists(npy_path):
                    os.makedirs(npy_path)

                np.save(os.path.join(npy_path, str(frame_num)), extract_keypoints(results))

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()
