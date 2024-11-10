import cv2  # Importar OpenCV para el procesamiento de imágenes y videos
import numpy as np  # Importar NumPy para operaciones numéricas y matrices
import mediapipe as mp  # Importar MediaPipe para la detección de landmarks
import tensorflow as tf  # Importar TensorFlow para cargar y utilizar el modelo de predicción

# Configuración de variables
sequence = []  # Lista para almacenar la secuencia de puntos clave detectados
sentence = []  # Lista para almacenar las predicciones que forman una oración
predictions = []  # Lista para almacenar las predicciones de la acción en cada frame
threshold = 0.4  # Umbral para la predicción de la acción, determinará la certeza mínima para aceptar una acción

actions = np.array(['F', 'J', 'S'])  # Array con las posibles acciones, en este caso las letras 'F', 'J', 'S'

# Cargar el modelo de Keras
model_path = '/Users/ashley/Desktop/Diseño e Innovación de Ingeniería 1/Tesis/Pruebas/PrimeraPrueba/Dinamicas/Propias/Prueba 2/action.h5'  # Ruta al modelo preentrenado en formato h5
model = tf.keras.models.load_model(model_path)  # Cargar el modelo desde el archivo especificado
tf.get_logger().setLevel('ERROR')  # Configurar el registro de TensorFlow para mostrar solo errores (sin advertencias o información)

# Configuración del modelo Mediapipe
mp_holistic = mp.solutions.holistic  # Holistic model de MediaPipe para la detección de pose, cara y manos
mp_drawing = mp.solutions.drawing_utils  # Utilidades para dibujar los landmarks detectados en la imagen

# Funciones auxiliares
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir la imagen de BGR (formato por defecto en OpenCV) a RGB (formato de MediaPipe)
    image.flags.writeable = False  # Desactivar la edición de la imagen para optimizar el procesamiento
    results = model.process(image)  # Procesar la imagen con el modelo de MediaPipe para obtener los resultados (landmarks)
    image.flags.writeable = True  # Volver a habilitar la edición de la imagen
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convertir la imagen de RGB de vuelta a BGR para que sea compatible con OpenCV
    return image, results  # Retornar la imagen procesada y los resultados obtenidos

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS)  # Dibujar los landmarks de la cara
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Dibujar los landmarks del cuerpo
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Dibujar los landmarks de la mano izquierda
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)  # Dibujar los landmarks de la mano derecha

def extract_keypoints(results):
    # Extraer los puntos clave de la pose (cuerpo), cara, mano izquierda y mano derecha y aplanarlos en un array
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])  # Concatenar todos los puntos clave en un solo array

# Iniciar captura de video y modelo Mediapipe
cap = cv2.VideoCapture(1)  # Iniciar la captura de video desde la cámara 1
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:  # Configurar el modelo Holistic con umbrales de confianza para detección y seguimiento
    while cap.isOpened():  # Mientras la cámara esté abierta y disponible para captura
        # Leer el frame de la cámara
        ret, frame = cap.read()  # Leer un frame de la cámara
        frame = cv2.flip(frame, 1)  # Voltear horizontalmente el frame para mejorar la visualización (como espejo)
        # Cambia el tamaño del frame antes de procesarlo
        frame = cv2.resize(frame, (540, 480))  # Redimensionar el frame a un tamaño menor (540x480) para mejorar el rendimiento

        if not ret:  # Si no se pudo leer el frame, salir del bucle
            break

        # Realizar detección y obtener resultados
        image, results = mediapipe_detection(frame, holistic)  # Llamar a la función de detección de MediaPipe

        # Extraer y almacenar keypoints en la secuencia
        keypoints = extract_keypoints(results)  # Extraer los puntos clave de los resultados de la detección
        sequence.append(keypoints)  # Añadir los puntos clave a la secuencia
        sequence = sequence[-30:]  # Mantener solo los últimos 30 conjuntos de puntos clave (para la predicción temporal)

        # 2. Lógica de predicción
        if len(sequence) == 30:  # Si la secuencia tiene 30 conjuntos de puntos clave (suficiente para hacer una predicción)
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]  # Hacer la predicción usando el modelo
            predictions.append(np.argmax(res))  # Añadir la acción predicha a la lista de predicciones
            predicted_action = actions[np.argmax(res)]  # Obtener la acción correspondiente a la predicción

            # Comprobación de estabilidad de predicción
            if np.unique(predictions[-15:])[0] == np.argmax(res):  # Verificar que las últimas 15 predicciones sean consistentes
                if res[np.argmax(res)] > threshold:  # Verificar si la confianza de la predicción supera el umbral
                    if len(sentence) > 0:  # Si ya hay predicciones previas en la oración
                        if predicted_action != sentence[-1]:  # Si la acción predicha es diferente a la última en la oración
                            sentence.append(predicted_action)  # Añadir la nueva acción a la oración
                    else:
                        sentence.append(predicted_action)  # Si no hay predicciones previas, añadir la acción inicial

            # Limitar longitud de la oración mostrada
            if len(sentence) > 5:  # Limitar la oración a 5 acciones para evitar que sea demasiado larga
                sentence = sentence[-5:]  # Mantener solo las últimas 5 acciones en la oración

        # Mostrar texto en pantalla
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)  # Dibujar un rectángulo de fondo para el texto
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Mostrar la oración formada en el frame

        # Mostrar imagen con inferencia en tiempo real
        cv2.imshow('OpenCV Feed', image)  # Mostrar el frame procesado con las predicciones en la ventana de OpenCV

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Esperar por una tecla, salir si se presiona 'q'
            break

# Liberar la cámara y cerrar las ventanas
cap.release()  # Liberar el dispositivo de captura de video
cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV abiertas
