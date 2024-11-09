import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Configuración de variables
sequence = []
sentence = []
predictions = []
threshold = 0.4

actions = np.array(['F', 'J', 'S'])

# Cargar el modelo de Keras
model_path = '/Users/ashley/Desktop/Diseño e Innovación de Ingeniería 1/Tesis/Pruebas/PrimeraPrueba/Dinamicas/Propias/Prueba 2/action.h5'  # Reemplaza con la ruta a tu modelo
model = tf.keras.models.load_model(model_path)
tf.get_logger().setLevel('ERROR')

# Configuración del modelo Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils  # Utilidades de dibujo

# Funciones auxiliares
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
    image.flags.writeable = False  # La imagen no es editable
    results = model.process(image)  # Predicción
    image.flags.writeable = True  # La imagen vuelve a ser editable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convertir de RGB a BGR
    return image, results


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


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


# Iniciar captura de video y modelo Mediapipe
cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Leer el frame de la cámara
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # Cambia el tamaño del frame antes de procesarlo
        frame = cv2.resize(frame, (540, 480))  # Define un tamaño menor, como (320, 240) o (640, 480)

        if not ret:
            break

        # Realizar detección y obtener resultados
        image, results = mediapipe_detection(frame, holistic)

        # Extraer y almacenar keypoints en la secuencia
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Mantiene solo los últimos 30 keypoints

        # 2. Lógica de predicción
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predictions.append(np.argmax(res))
            predicted_action = actions[np.argmax(res)]

            # Comprobación de estabilidad de predicción
            if np.unique(predictions[-15:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if predicted_action != sentence[-1]:
                            sentence.append(predicted_action)
                    else:
                        sentence.append(predicted_action)

            # Limitar longitud de la oración mostrada
            if len(sentence) > 5:
                sentence = sentence[-5:]

        # Mostrar texto en pantalla
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Mostrar imagen con inferencia en tiempo real
        cv2.imshow('OpenCV Feed', image)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()