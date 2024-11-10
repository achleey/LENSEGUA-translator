import pickle        # Para serializar objetos
import tensorflow as tf  # Asegúrate de tener TensorFlow Lite instalado
import cv2        # Para procesamiento de imágenes y captura de video        
import mediapipe as mp        # Para detección de landmarks en imágenes
import numpy as np        # Para manejo de arrays y operaciones numéricas

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

# Función para extraer puntos clave de las manos, cara y cuerpo
def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 2)  # 33 landmarks, 2 coords (x, y)
    face = np.array([[res.x, res.y] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 2)  # 468 landmarks, 2 coords (x, y)
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 2)  # 21 landmarks, 2 coords (x, y)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 2)  # 21 landmarks, 2 coords (x, y)
    return np.concatenate([pose, face, lh, rh])        # Concatenamos todos los puntos clave en un solo arreglo

# Inicialización de la cámara (con índice 1 para la segunda cámara)
cap = cv2.VideoCapture(1)
sequence = []        # Secuencia de puntos clave
frame_count = 0
predictions = []

while True:

    ret, frame = cap.read()        # Leer un frame de la cámara
    frame = cv2.flip(frame, 1)        # Voltear la imagen horizontalmente para obtener un efecto espejo
    H, W, _ = frame.shape        # Obtener las dimensiones de la imagen
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        # Convertir la imagen de BGR a RGB para Mediapipe
    results = holistic.process(frame_rgb)        # Procesar el frame para detectar puntos clave
    keypoints = extract_keypoints(results)        # Extraer los puntos clave de la imagen
    sequence.append(keypoints)        # Agregar los puntos clave a la secuencia
    sequence = sequence[-30:]        # Mantener solo los últimos 30 frames

    # Hacer la predicción cada vez que tengamos 30 frames en la secuencia
    if len(sequence) == 30 and frame_count % 1 == 0: # Se hace la predicción cada 30 segundos.
        # Preparar los datos de entrada para el modelo
        input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)        # Cargar los datos en el modelo
        interpreter.invoke()        # Ejecutar el modelo

        # Obtener la salida del modelo y predecir la clase
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_class_index = np.argmax(prediction)        # Obtener el índice de la clase con mayor probabilidad
        predicted_character = labels_dict[predicted_class_index]        # Convertir el índice en la letra correspondiente
        print(f"Predicción: {predicted_character} - Confianza: {prediction[predicted_class_index]}")
    else:
        predicted_character = 'No se detecta una combinación válida'        # Mensaje si la secuencia no está completa
        print("Esperando secuencia completa para predecir...")

    # Extraer las coordenadas de las manos (izquierda y derecha)
    lh = keypoints[33 * 2 + 468 * 2:33 * 2 + 468 * 2 + 21 * 2]
    rh = keypoints[33 * 2 + 468 * 2 + 21 * 2:]

    # Inicializamos las listas para las coordenadas de las manos
    hand_data = {'left_hand': [], 'right_hand': []}

    if lh.size > 0:
        hand_data['left_hand'] = lh        # Si hay puntos clave de la mano izquierda, agregar al diccionario

    if rh.size > 0:
        hand_data['right_hand'] = rh        # Si hay puntos clave de la mano derecha, agregar al diccionario

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

    # Mostrar el frame con las predicciones en la ventana
    cv2.imshow('Inference', frame)

    frame_count += 1        # Contador de frames procesados

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()        # Liberar la cámara
cv2.destroyAllWindows()        # Cerrar todas las ventanas de OpenCV
