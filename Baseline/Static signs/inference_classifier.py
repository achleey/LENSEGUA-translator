import pickle                  # Para guardar y cargar datos en archivos binarios.
import cv2                     # Biblioteca para detectar y procesar poses y landmarks
import mediapipe as mp         # Opencv para procesamiento de imágenes
import numpy as np             # Para operaciones con arrays númericos

# Cargar el modelo de reconocimiento de signos
model_dict = pickle.load(open('./model.p', 'rb'))  # Archivo de modelo preentrenado en formato binario
model = model_dict['model']

cap = cv2.VideoCapture(1)      # Captura de vídeo desde la cámara

# Configuración de MediaPipe para detección y dibujo de manos
mp_hands = mp.solutions.hands  
mp_drawing = mp.solutions.drawing_utils  
mp_drawing_styles = mp.solutions.drawing_styles  

# Modelo de detección con confianza mínima
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) 

labels_dict = {0: 'A', 1: 'B', 2: 'C'}  # Etiquetas para la clasificación de cada seña

while True:
    data_aux = []    # Listas auxiliares para coordenadas de landmarks de la mano
    x_ = []
    y_ = []

    ret, frame = cap.read()    # Captura cada frame de la cámara
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir a RGB para usar con MediaPipe

    # Procesar detección de manos 
    results = hands.process(frame_rgb) 
    if results.multi_hand_landmarks:    # Si hay detección
        for hand_landmarks in results.multi_hand_landmarks:  
           # Dibujar landmarks y conexiones en el frame
            mp_drawing.draw_landmarks(  
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Extraer coordenadas x, y de cada landmark de la mano detectada
        for hand_landmarks in results.multi_hand_landmarks:  
            for i in range(len(hand_landmarks.landmark)):  
                x = hand_landmarks.landmark[i].x  # Coordenadas x e y normalizadas
                y = hand_landmarks.landmark[i].y  
                data_aux.append(x)  # Agregar coordenadas al arreglo data_aux
                data_aux.append(y)  
                x_.append(x)
                y_.append(y)

       # Obtener coordenadas mínimas y máximas para encuadrar la mano detectada 
        x1 = int(min(x_) * W)  
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

       # Predecir el carácter correspondiente usando el modelo cargado
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Mostrar el carácter predicho y un rectángulo alrededor de la mano
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(frame, predicted_character, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2,
                    cv2.LINE_AA) 

   # Mostrar el frame procesado
    cv2.imshow('Frame', frame)
    cv2.waitKey(25)

# Liberar la captura de vídeo y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
