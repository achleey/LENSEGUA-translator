import os        # Proporciona funciones para interactuar con el sistema operativo
import pickle        # Permite la serialización y deserialización de objetos en Python
import tensorflow as tf        # Biblioteca de aprendizaje profundo para cargar modelos en formato TensorFlow Lite
import time        # Biblioteca para medir el tiempo de ejecución
import cv2        # Biblioteca OpenCV para procesamiento de imágenes y video
import numpy as np        # Biblioteca para operaciones matemáticas y manipulaciones de matrices
import mediapipe as mp        # Biblioteca para el procesamiento de pose, manos y rostro en video

# Importación de módulos de PyQt6 para la creación de interfaces gráficas
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QGraphicsScene, QGraphicsView, \
    QGraphicsTextItem, QGraphicsPixmapItem, QGraphicsProxyWidget
from PyQt6.QtGui import QPixmap, QFont, QIcon, QPen, QColor, QImage        # Módulos de PyQt6 para gestionar gráficos y estilos
from PyQt6.QtCore import Qt, QPropertyAnimation, QRect, QRectF, QThread, pyqtSignal, QUrl        # Módulos para propiedades y animaciones
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput        # Módulos multimedia de PyQt6 para audio y video

# Clase que maneja el hilo de captura de video
class VideoThread(QThread):
    frame_captured = pyqtSignal(QImage)        # Señal para capturar el frame
    result_captured = pyqtSignal(str)  # Señal para capturar el resultado de la predicción

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(1)        # Inicialización de la captura de video (cámara)
        self.running = True        # Bandera de control para el hilo
        self.sequence = []        # Secuencia de frames para el análisis de acción dinámica
        self.sequence_length = 30  # Longitud de la secuencia para señas dinámicas
        self.frame_count = 0        # Contador de frames

        # Variables para el control de visibilidad de landmarks de manos, rostro y pose
        self.show_hands = False
        self.show_face = False
        self.show_pose = False

        # Inicializar mediapipe
        # Manos
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

        # Cara
        self.mp_face = mp.solutions.face_mesh
        self.face = self.mp_face.FaceMesh()

        # Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3)

        # Holistico
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)
        
        self.mp_drawing = mp.solutions.drawing_utils        # Utilidades para dibujar landmarks
        self.mp_drawing_styles = mp.solutions.drawing_styles        # Estilos de dibujo

        # Cargar modelos y diccionarios de etiquetas
        self.model1 = pickle.load(open('./Model1HandV.p', 'rb'))['model']
        self.model2 = pickle.load(open('./Model2HandsV.p', 'rb'))['model']
        self.model3 = pickle.load(open('./ModelHandAndFaceV.p', 'rb'))['model']
        self.model4 = pickle.load(open('./ModelHandAndBodyV.p', 'rb'))['model']

        # Cargar el modelo de TensorFlow Lite
        self.interpreter = tf.lite.Interpreter(model_path="./actionV.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Diccionario de etiquetas
        self.labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'K', 6: 'L', 7: 'M', 8: 'N', 9: 'O', 10: 'P', 11: 'R', 12: 'U', 13: 'V', 14: 'W', 15: 'Y'}
        self.labels_dict2 = {0: 'Ñ', 1: 'Q', 2: 'X'}
        self.labels_dict3 = {0: 'G', 1: 'H', 2: 'I', 3: 'T'}
        self.labels_dict4 = {0: 'Z'}
        self.labels_dict_dynamics = {0: 'F', 1: 'J', 2: 'S'}

    # Función para comprobar si la mano está cerca del rostro
    def is_hand_near_face(self, hand_landmarks, face_landmarks, threshold=0.03):
        for hand_point in hand_landmarks:
            for face_point in face_landmarks:
                distance = np.sqrt((hand_point.x - face_point.x) ** 2 + (hand_point.y - face_point.y) ** 2)
                if distance < threshold:
                    return True
        return False

    # Función para verificar si la mano está horizontal
    def is_hand_horizontal(self,hand_landmarks, angle_threshold=45):
        if not hand_landmarks:
            return False

        landmarks = hand_landmarks.landmark
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        index_finger_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]

        # Calcular el ángulo de la mano respecto al eje horizontal
        dx = index_finger_tip.x - pinky_tip.x
        dy = index_finger_tip.y - pinky_tip.y
        angle = np.arctan2(dy, dx) * 180 / np.pi

        return np.abs(angle) < angle_threshold or np.abs(angle - 180) < angle_threshold

    # Función para verificar si el codo es visible
    def is_elbow_visible(self, pose_landmarks, hand_landmarks, pose_threshold=0.5, hand_threshold=45):
        if not pose_landmarks or not hand_landmarks:
            return False

        # Verificar si el codo es visible
        landmarks = pose_landmarks.landmark
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]

        left_elbow_visible = left_elbow.visibility > pose_threshold
        right_elbow_visible = right_elbow.visibility > pose_threshold

        # Verificar si la mano está en posición horizontal
        hand_horizontal = self.is_hand_horizontal(hand_landmarks, angle_threshold=hand_threshold)

        return (left_elbow_visible or right_elbow_visible) and hand_horizontal

    # Función para extraer keypoints de los resultados de mediapipe
    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 2)
        face = np.array([[res.x, res.y] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 2)
        lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 2)
        rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 2)
        return np.concatenate([pose, face, lh, rh])

    # Función para predecir la acción dinámica con TensorFlow Lite
    def predict_dynamic_action(self):
        # Convertir secuencia a formato de entrada para TensorFlow Lite
        input_data = np.expand_dims(self.sequence, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        predicted_class_index = np.argmax(prediction)
        predicted_label = self.labels_dict_dynamics[predicted_class_index]
        return predicted_label

    # Función para verificar si la seña dinámica está activa
    def is_dynamic_sign_active(self, threshold=0.020, keypoints_to_track=None):
        # Asegurarse de que hay suficientes frames para detectar movimiento
        if len(self.sequence) < 2:
            return False

        # Selección de keypoints a rastrear (si no se especifican)
        keypoints_to_track = [0, 4, 8, 12, 16, 20, 12, 13, 15, 16]

        total_movement = 0.0

        for i in range(1, len(self.sequence)):
            frame_movement = 0.0  # Movimiento acumulado en este frame

            # Calcular movimiento entre frames consecutivos para los keypoints seleccionados
            for k in keypoints_to_track:
                if k < len(self.sequence[i]) and k < len(self.sequence[i - 1]):
                    movement = np.linalg.norm(
                        np.array(self.sequence[i][k]) - np.array(self.sequence[i - 1][k])
                    )
                    frame_movement += movement

            total_movement += frame_movement

        # Calcular promedio de movimiento
        average_movement = total_movement / (len(self.sequence) - 1)

        return average_movement > threshold

    def run(self):
        # Bucle principal para la captura de frames mientras el hilo está en ejecución
        while self.running:
            ret, frame = self.cap.read()        # Captura un frame de la cámara
            if ret:
                frame = cv2.flip(frame, 1)        # Flip para mostrar la imagen en modo espejo
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        # Conversión a RGB

                # Procesar diferentes resultados con mediapipe
                results_hands = self.hands.process(frame_rgb)
                results_face = self.face.process(frame_rgb)
                results_pose = self.pose.process(frame_rgb)
                results_holistic = self.holistic.process(frame_rgb)

                # Variables auxiliares para guardar la información de las detecciones
                num_hands = 0
                face_detected = False
                face_data_aux = []
                hand_data_aux = []
                pose_data_aux = []

                # Procesar las manos
                if results_hands.multi_hand_landmarks:
                    num_hands = len(results_hands.multi_hand_landmarks)
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        hand_data_aux = []
                        for landmark in hand_landmarks.landmark:
                            xh = landmark.x
                            yh = landmark.y
                            hand_data_aux.append(xh)        # Guardar coordenadas de la mano
                            hand_data_aux.append(yh)

                # Procesar la cara
                if results_face.multi_face_landmarks:
                    face_detected = True
                    for face_landmarks in results_face.multi_face_landmarks:
                        for landmark in face_landmarks.landmark:
                            xf = landmark.x
                            yf = landmark.y
                            face_data_aux.append(xf)        # Guardar coordenadas de la cara
                            face_data_aux.append(yf)

                # Procesar la pose
                if results_pose.pose_landmarks:
                    pose_data = []
                    pose_data_aux = []
                    landmarks = results_pose.pose_landmarks.landmark
                    pose_data_aux.append(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x)
                    pose_data_aux.append(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y)
                    pose_data_aux.append(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x)
                    pose_data_aux.append(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y)
                    pose_data_aux.append(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x)
                    pose_data_aux.append(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y)
                    pose_data_aux.append(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
                    pose_data_aux.append(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y)

                # Variable para la predicción
                predicted_character = 'Traducción...'

                keypoints = self.extract_keypoints(results_holistic)  # Extraer keypoints del frame actual

                # Verificar si hay landmarks en las manos en el frame actual
                hand_present = results_holistic.left_hand_landmarks or results_holistic.right_hand_landmarks

                # Solo agregamos los keypoints si hay landmarks de las manos presentes
                if hand_present:
                    self.sequence.append(keypoints)

                self.sequence = self.sequence[-30:]        # Mantener los últimos 30 frames

                # Lógica de predicción basada en las manos y la cara detectadas
                if num_hands >= 1 and face_detected:
                    if self.is_hand_near_face(results_hands.multi_hand_landmarks[0].landmark, results_face.multi_face_landmarks[0].landmark):

                        if self.is_dynamic_sign_active():
                            if len(self.sequence) == 30 and self.frame_count % 1 == 0:  # Se hace la predicción cada 30 segundos.
                                predicted_character = self.predict_dynamic_action()  # Predecir la acción dinámica
                        else:
                            # Combina datos de mano y cara para hacer la predicción
                            combination = hand_data_aux + face_data_aux
                            prediction = self.model3.predict([np.asarray(combination)])
                            predicted_character = self.labels_dict3[int(prediction[0])]
                    else:
                        # Predicción basada solo en la mano detectada
                        if num_hands == 1:        # Si hay una mano
                            if self.is_dynamic_sign_active():
                                if len(self.sequence) == 30 and self.frame_count % 1 == 0:  # Se hace la predicción cada 30 segundos.
                                    predicted_character = self.predict_dynamic_action()  # Predecir la acción dinámica
                            else:
                                prediction = self.model1.predict([np.asarray(hand_data_aux)])
                                predicted_character = self.labels_dict[int(prediction[0])]
                        else:        # Si hay dos manos
                            if self.is_dynamic_sign_active():
                                if len(self.sequence) == 30 and self.frame_count % 1 == 0:  # Se hace la predicción cada 30 segundos.
                                    predicted_character = self.predict_dynamic_action()  # Predecir la acción dinámica
                            else:
                                prediction = self.model2.predict([np.asarray(hand_data_aux)])
                                predicted_character = self.labels_dict2[int(prediction[0])]

                # Predicción basada en la visibilidad del codo
                if num_hands == 1 and self.is_elbow_visible(results_pose.pose_landmarks, results_hands.multi_hand_landmarks[0]):
                    combinationByH = hand_data_aux + pose_data_aux
                    prediction = self.model4.predict([np.asarray(combinationByH)])
                    predicted_character = self.labels_dict4[int(prediction[0])]

                self.frame_count += 1        # Incrementar el contador de frames
                self.result_captured.emit(predicted_character)        # Emitir el resultado de la predicción

                # Detección de manos
                if self.show_hands:
                    if results_hands.multi_hand_landmarks:
                        for hand_landmarks in results_hands.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                frame_rgb,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )

                # Detección de cara
                if self.show_face:
                    if results_face.multi_face_landmarks:
                        for face_landmarks in results_face.multi_face_landmarks:
                            self.mp_drawing.draw_landmarks(
                                frame_rgb,
                                face_landmarks,
                                self.mp_face.FACEMESH_CONTOURS,
                                self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                                self.mp_drawing_styles.get_default_face_mesh_contours_style()
                            )

                # Detección de pose
                if self.show_pose:
                    results_pose = self.pose.process(frame_rgb)
                    if results_pose.pose_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame_rgb,
                            results_pose.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            self.mp_drawing_styles.get_default_pose_landmarks_style()
                        )

                # Convertir el frame procesado a formato Qt y emitir
                h, w, ch = frame_rgb.shape
                qt_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.frame_captured.emit(qt_image)

    def stop(self):
        # Detener la captura de video y liberar recursos
        self.running = False
        self.cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Configurar la ventana principal
        self.setWindowTitle("LENSEGUAtraductor")
        self.setGeometry(100, 100, 1280, 720)        # Tamaño y posición de la ventana
        self.setStyleSheet("background-color: #DBE2EA;")        # Estilo de fondo

        # Widget central y layout
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)        # Layout vertical para los elementos
        layout.setContentsMargins(0, 0, 0, 0)        # Sin márgenes
        self.setCentralWidget(central_widget)        # Establecer el widget central

        # Crear una escena y vista para el canvas
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)        # Vista para mostrar la escena
        layout.addWidget(self.view)        # Agregar vista al layout

        # Desactivar barras de desplazamiento
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Asegurarse de que la vista esté alineada correctamente
        self.view.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # Agregar un rectángulo a la escena
        self.scene.addRect(
            0.0, 0.0, 1280.0, 60.0,
            pen=QPen(Qt.PenStyle.NoPen),        # Sin borde
            brush=QColor("#FFFFFF")        # Fondo blanco
        )

        # Encabezado
        self.add_text("LENSEGUAtraductor", 16.0, 8.0, 28, QColor("#000000"))

        # Agregar imagen de fondo para el frame de inferencia
        self.add_image("image_4.png", 50.5, 109.8)

        # Obtener el tamaño de la imagen "image_4.png"
        self.image_4_pixmap = QPixmap("image_4.png")
        image_4_size = self.image_4_pixmap.size()

        # Crear el QGraphicsPixmapItem para el video
        self.frame_item = QGraphicsPixmapItem()
        self.frame_item.setZValue(10)  # Asegúrate de que esté en la capa superior
        self.scene.addItem(self.frame_item)  # Añadir a la escena

        # Establecer tamaño y posición iniciales para el frame_item
        self.frame_item.setPos(96.5, 145)  # Ajustar la posición según sea necesario # MODIFICAR COORDENADAS PARA FRAME
        self.frame_item.setPixmap(QPixmap(image_4_size))  # Establecer tamaño inicial basado en image_4

        # Área de traducción
        self.add_image("image_2.png", 738.0, 313.8)
        self.add_text("Traducción a texto", 879.0, 357.8, 28, QColor("#000000"))
        self.add_image("image_3.png", 863.0, 407.8)
        self.translation_text_item = self.add_text("", 883.0, 424.8, 16, QColor("#1F285B"), bold=False)

        # Opciones de traducción
        self.add_image("image_1.png", 738, 126.8)
        self.add_text("Opciones de traducción", 850.0, 170.8, 28, QColor("#000000"))

        # Crear botones con animaciones
        self.text_button, anim1 = self.create_button("button_1.png", 889.5, 230.8, 90.0, 59.0, self.text_button_clicked)
        self.audio_button, anim2 = self.create_button("button_2.png", 1038.0, 230.8, 90.0, 59.0, self.audio_button_clicked)

        # Añadir los botones a la escena usando QGraphicsProxyWidget
        self.add_widget_to_scene(self.text_button, 889.5, 230.8)
        self.add_widget_to_scene(self.audio_button, 1038.0, 230.8)

        # Guardar animaciones para acceso futuro
        self.animations = {self.text_button: anim1, self.audio_button: anim2}

        # Habilitar los botones
        self.text_button.setEnabled(True)
        self.audio_button.setEnabled(True)

        # Guardar las imágenes y el estado de los botones
        self.text_button_images = ("button_1.png", "colored_button_1.png")
        self.audio_button_images = ("button_2.png", "colored_button_2.png")

        # Bandera para indicar si se debe actualizar el texto y el audio
        self.update_text = False
        self.update_audio = False

        # Inicializar QMediaPlayer para reproducción de audio
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        # Landmarks
        self.add_text("Landmarks - manos", 80.0, 552.0, 16, QColor("#000000"))
        self.add_text("Landmarks - cara", 80.0, 602.0, 16, QColor("#000000"))
        self.add_text("Landmarks - pose", 80.0, 652.0, 16, QColor("#000000"))

        # Inicializar la lista de switches
        self.switches = []

        # Crear interruptores
        self.create_switch(275, 557, 'hands')
        self.create_switch(275, 607, 'face')
        self.create_switch(275, 657, 'pose')

        # Inicializar y comenzar el hilo de video
        self.video_thread = VideoThread()
        self.video_thread.frame_captured.connect(self.update_frame)
        self.video_thread.result_captured.connect(self.handle_result_captured)  # Conectar a la función manejadora
        self.video_thread.start()

    # Método para agregar texto a la escena
    def add_text(self, text, x, y, font_size, color, bold=True):
        text_item = QGraphicsTextItem(text)
        font = QFont("IBM Plex Sans", font_size, QFont.Weight.Bold if bold else QFont.Weight.Normal)
        text_item.setFont(font)
        text_item.setDefaultTextColor(color)
        text_item.setPos(x, y)
        self.scene.addItem(text_item)
        return text_item

    # Método para agregar imágenes a la escena
    def add_image(self, image_path, x, y):
        pixmap = QPixmap(image_path)
        item = QGraphicsPixmapItem(pixmap)
        item.setOffset(x, y)
        self.scene.addItem(item)

    # Crear botón con imagen, animación y acción de clic
    def create_button(self, image_path, x, y, width, height, callback):
        button = QPushButton()

        pixmap = QPixmap(image_path)
        icon = QIcon(pixmap)
        button.setIcon(icon)
        button.setIconSize(pixmap.size())
        button.setFixedSize(pixmap.size())
        button.setFlat(True)
        button.setStyleSheet("border: none; padding: 0px; margin: 0px;")
        button.clicked.connect(callback)
        button.setGeometry(int(x), int(y), int(width), int(height))

        animation = QPropertyAnimation(button, b"geometry")
        animation.setDuration(100)
        animation.setStartValue(QRect(int(x), int(y), int(width), int(height)))
        animation.setEndValue(QRect(int(x) + 5, int(y) + 5, int(width) - 10, int(height) - 10))

        return button, animation

    # Cambiar la imagen del botón
    def set_button_image(self, button, image_path):
        pixmap = QPixmap(image_path)
        icon = QIcon(pixmap)
        button.setIcon(icon)
        button.setIconSize(pixmap.size())
        button.setFixedSize(pixmap.size())

    # Acción cuando se hace clic en el botón de texto
    def text_button_clicked(self):
        print("Text button clicked")

        if not self.update_text:
            self.set_button_image(self.text_button, self.text_button_images[1])
            self.set_button_image(self.audio_button, self.audio_button_images[0])

            self.animate_button(self.text_button)
            self.update_text = True  # Activar la actualización del texto
            self.update_audio = False

    # Acción cuando se hace clic en el botón de audio
    def audio_button_clicked(self):
        print("Audio button clicked")

        if not self.update_audio:
            self.set_button_image(self.audio_button, self.audio_button_images[1])
            self.set_button_image(self.text_button, self.text_button_images[0])

            self.animate_button(self.audio_button)
            self.update_text = False  # Desactivar la actualización del texto
            self.update_audio = True

     # Manejar la traducción capturada
    def handle_result_captured(self, translation_text):
        if self.update_text:
            self.update_translation(translation_text)
        if self.update_audio:
            self.play_audio(translation_text)

    # Actualizar la traducción en el área de texto
    def update_translation(self, translation_text):
        self.translation_text_item.setPlainText(translation_text)

    # Reproducir el audio correspondiente a la traducción
    def play_audio(self, translation_text):

        if translation_text in ['Traducción...']:
            print(f"Ignorando valor no deseado: {translation_text}")
            return

        audio_path = os.path.abspath(f"./audio/{translation_text}.mp3")
        print(f"Intentando reproducir: {audio_path}")  # Imprime la ruta completa del archivo para verificar
        if os.path.exists(audio_path):
            url = QUrl.fromLocalFile(audio_path)
            self.media_player.setSource(url)  # Usa setSource aquí directamente
            self.media_player.play()
            print("Reproducción iniciada")
        else:
            print(f"Archivo no encontrado: {audio_path}")

    # Animación de los botones
    def animate_button(self, button):
        animation = self.animations[button]
        animation.setDirection(QPropertyAnimation.Direction.Forward)
        animation.finished.connect(lambda: self.reset_button_geometry(button))
        animation.start()

    # Restablece la geometría original del botón con animación hacia atrás
    def reset_button_geometry(self, button):
        animation = self.animations[button]
        animation.finished.disconnect()        # Desconecta cualquier conexión previa
        animation.setDirection(QPropertyAnimation.Direction.Backward)        # Dirección de la animación hacia atrás
        animation.start()        # Inicia la animación
        animation.finished.connect(lambda: self.restore_button_position(button))        # Restaura la posición después de la animación

    # Restaura la geometría original del botón
    def restore_button_position(self, button):
        animation = self.animations[button]
        animation.finished.disconnect()
        original_geometry = animation.startValue()        # Obtiene la geometría original de la animación
        button.setGeometry(original_geometry)        # Restaura la geometría del botón

    # Crea un interruptor con imágenes para los estados "on" y "off"
    def create_switch(self, x, y, switch_type):
        switch_on = QPixmap("switch_on.png")
        switch_off = QPixmap("switch_off.png")

        button_switch = QPushButton()        # Crea un botón de tipo interruptor
        button_switch.setIcon(QIcon(switch_off))        # Establece el ícono inicial (apagado)
        button_switch.setIconSize(switch_off.size())        # Ajusta el tamaño del ícono
        button_switch.setFixedSize(switch_off.size())        # Establece un tamaño fijo para el botón
        button_switch.setFlat(True)        # Elimina bordes del botón
        button_switch.setStyleSheet("border: none; padding: 0px; margin: 0px; background: transparent;")        # Estilo sin bordes ni fondo

        button_switch.switch_state = False  # Estado inicial del interruptor (apagado)
        button_switch.switch_on = switch_on        # Icono del interruptor encendido
        button_switch.switch_off = switch_off        # Icono del interruptor apagado
        button_switch.clicked.connect(lambda: self.switch(button_switch, switch_type))        # Conecta el clic a la acción de cambio de estado

        self.switches.append(button_switch)        # Añade el interruptor a la lista de switches
        self.add_widget_to_scene(button_switch, x, y)        # Añade el interruptor a la escena en la posición (x, y)

    # Añade el widget a la escena en las coordenadas dadas
    def add_widget_to_scene(self, widget, x, y):
        proxy = QGraphicsProxyWidget()        # Crea un proxy para el widget
        proxy.setWidget(widget)        # Establece el widget en el proxy
        proxy.setPos(x, y)        # Establece la posición del proxy
        widget_size = widget.size()        # Obtiene el tamaño del widget
        proxy.resize(widget_size.width(), widget_size.height())        # Redimensiona el proxy para que coincida con el tamaño del widget
        self.scene.addItem(proxy)        # Añade el proxy a la escena

    # Cambia el estado del interruptor entre "on" y "off"
    def switch(self, button_switch, switch_type):
        if button_switch.switch_state:
            button_switch.setIcon(QIcon(button_switch.switch_off))        # Cambia el ícono a "off"
            button_switch.switch_state = False        # Establece el estado a "off"
        else:
            button_switch.setIcon(QIcon(button_switch.switch_on))        # Cambia el ícono a "on"
            button_switch.switch_state = True        # Establece el estado a "on"

        self.update_switch_size(button_switch)        # Actualiza el tamaño del interruptor según su estado
        self.toggle_switch(switch_type)        # Llama a la función que maneja la lógica del tipo de interruptor

    # Actualiza el tamaño del interruptor según su estado
    def update_switch_size(self, button_switch):
        if button_switch.switch_state:
            size = button_switch.switch_on.size()        # Obtiene el tamaño del ícono "on"
        else:
            size = button_switch.switch_off.size()        # Obtiene el tamaño del ícono "off"

        button_switch.setFixedSize(size)        # Establece el tamaño fijo del interruptor

        # Ajusta la geometría del proxy manteniendo su posición
        for item in self.scene.items():
            if isinstance(item, QGraphicsProxyWidget) and item.widget() == button_switch:
                # Obtener la posición actual del item
                current_pos = item.pos()
                # Ajustar la geometría manteniendo la posición actual
                item.setGeometry(QRectF(0, 0, size.width(), size.height()))
                item.setPos(current_pos)  # Mantener la posición actual
                break

    # Cambia el estado del interruptor correspondiente a manos, cara o pose
    def toggle_switch(self, switch_type):
        if switch_type == 'hands':
            self.video_thread.show_hands = not self.video_thread.show_hands
        elif switch_type == 'face':
            self.video_thread.show_face = not self.video_thread.show_face
        elif switch_type == 'pose':
            self.video_thread.show_pose = not self.video_thread.show_pose

    # Actualiza el frame mostrado con una nueva imagen escalada
    def update_frame(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)

        # Obtener el tamaño de la imagen "image_4.png"
        image_4_size = self.image_4_pixmap.size()

        # Definir el tamaño deseado para la imagen del frame
        desired_width = image_4_size.width() -90 # Ejemplo: reducir el ancho en 20 píxeles
        desired_height = image_4_size.height() - 90 # Ejemplo: reducir la altura en 20 píxeles

        # Ajustar el pixmap para que coincida con el tamaño deseado
        scaled_pixmap = pixmap.scaled(desired_width, desired_height, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)

        # Establecer el pixmap escalado en el frame_item
        self.frame_item.setPixmap(scaled_pixmap)

        # Ajustar la posición si es necesario
        self.frame_item.setPos(96.5, 145)  # Asegúrate de que la posición sea la correcta # MODIFICAR COORDENADAS PARA FRAME

    # Maneja el evento de cierre de la ventana
    def closeEvent(self, event):
        self.video_thread.stop()
        self.video_thread.wait()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication([])        # Crea la aplicación
    window = MainWindow()        # Crea la ventana principal
    window.show()        # Muestra la ventana
    app.exec()        # Ejecuta el ciclo de eventos de la aplicación
