import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))  # Se carga el archivo model.pickle. Se indica r para leer y b en binario
model = model_dict['model']

cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands  # Se importa el modulo hands del paquete solutions. Para detectar manos
mp_drawing = mp.solutions.drawing_utils  # Importar drawing_utils de solutions. Para dibujar sobre imgs y videos.
mp_drawing_styles = mp.solutions.drawing_styles  # Importar drawing_styles de solutions. Para personalizar apariencia de dibujos.

hands = mp_hands.Hands(static_image_mode=True,
                       min_detection_confidence=0.3)  # Modelo de deteccion de manos, estaticas. 1 altamente confiable, 0 no confiable

labels_dict = {0: 'A', 1: 'B', 2: 'C'}  # Etiquetas para la predicción

while True:

    data_aux = []  # Listas auxiliares para procesamiento de landmarks
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # se cambia de BGR a RGB para poder usar con matplotlib y mediapipe

    results = hands.process(frame_rgb)  # .process procesa la imagen img_rgb con el modelo de deteccion hands.
    if results.multi_hand_landmarks:  # verifica que se detecten manos en results. multi_hand... es una lista con puntos clave de las manos detectadas
        for hand_landmarks in results.multi_hand_landmarks:  # itera sobre cada conjunto de puntos claves de la lista de cada mano.
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:  # itera sobre cada conjunto de puntos claves de la lista de cada mano.
            for i in range(len(hand_landmarks.landmark)):  # itera sobre los conjuntos de puntos clave de la mano actual
                x = hand_landmarks.landmark[i].x  # Acceder a coordenada x de conjunto de datos i
                y = hand_landmarks.landmark[i].y  # Acceder a coordenada y de conjunto de datos i
                data_aux.append(x)  # Agrega coordenada x al arreglo data_aux
                data_aux.append(y)  # Agrega coordenada y al arreglo data_aux
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W)  # Dibuja la predicción alrededor de la mano
        y1 = int(min(y_) * H)

        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(frame, predicted_character, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2,
                    cv2.LINE_AA)  # texto para indicar al usuario que inicie el proceso de captura

    cv2.imshow('Frame', frame)
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()
