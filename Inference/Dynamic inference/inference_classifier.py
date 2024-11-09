import pickle
import tensorflow as tf  # Asegúrate de tener TensorFlow Lite instalado
import cv2
import mediapipe as mp
import numpy as np

# Carga del modelo TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path="./actionV.tflite")
interpreter.allocate_tensors()

# Obtiene los detalles de los tensores de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Diccionario de etiquetas
labels_dict = {0: 'F', 1: 'J', 2: 'S'}

# Configuración de Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 2)  # 33 landmarks, 2 coords (x, y)
    face = np.array([[res.x, res.y] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 2)  # 468 landmarks, 2 coords (x, y)
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 2)  # 21 landmarks, 2 coords (x, y)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 2)  # 21 landmarks, 2 coords (x, y)
    return np.concatenate([pose, face, lh, rh])

cap = cv2.VideoCapture(1)
sequence = []
frame_count = 0
predictions = []

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    if len(sequence) == 30 and frame_count % 1 == 0: # Se hace la predicción cada 30 segundos.
        # Preparar los datos de entrada para el modelo
        input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Obtener la salida del modelo y predecir la clase
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_class_index = np.argmax(prediction)
        predicted_character = labels_dict[predicted_class_index]
        print(f"Predicción: {predicted_character} - Confianza: {prediction[predicted_class_index]}")
    else:
        predicted_character = 'No se detecta una combinación válida'
        print("Esperando secuencia completa para predecir...")

    # Dibujar el cuadro alrededor de la mano
    lh = keypoints[33 * 2 + 468 * 2:33 * 2 + 468 * 2 + 21 * 2]
    rh = keypoints[33 * 2 + 468 * 2 + 21 * 2:]

    # Inicializamos las listas para las coordenadas de las manos
    hand_data = {'left_hand': [], 'right_hand': []}

    if lh.size > 0:
        hand_data['left_hand'] = lh

    if rh.size > 0:
        hand_data['right_hand'] = rh

    # Dibujar los cuadros alrededor de las manos si están presentes
    for hand, coords in hand_data.items():
        if len(coords) > 0:
            coords = np.array(coords).flatten()
            x1 = int(min(coords[::2]) * W)  # Coordenadas X mínimas
            y1 = int(min(coords[1::2]) * H)  # Coordenadas Y mínimas
            x2 = int(max(coords[::2]) * W)  # Coordenadas X máximas
            y2 = int(max(coords[1::2]) * H)  # Coordenadas Y máximas

            # Dibujar el cuadro alrededor de la mano
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 5)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (255, 255, 255), 3, cv2.LINE_AA)

    # Mostrar el carácter predicho en la esquina superior izquierda si no hay manos detectadas
    if not any(len(coords) > 0 for coords in hand_data.values()):
        cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.13,
                    (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Inference', frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()