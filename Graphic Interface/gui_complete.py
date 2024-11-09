import os
import pickle
import tensorflow as tf
import time
import cv2
import numpy as np
import mediapipe as mp


from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QGraphicsScene, QGraphicsView, \
    QGraphicsTextItem, QGraphicsPixmapItem, QGraphicsProxyWidget
from PyQt6.QtGui import QPixmap, QFont, QIcon, QPen, QColor, QImage
from PyQt6.QtCore import Qt, QPropertyAnimation, QRect, QRectF, QThread, pyqtSignal, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

class VideoThread(QThread):
    frame_captured = pyqtSignal(QImage)
    result_captured = pyqtSignal(str)  # Añadir esta línea para definir la señal result_captured

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(1)
        self.running = True
        self.sequence = []
        self.sequence_length = 30  # Longitud de la secuencia para señas dinámicas
        self.frame_count = 0

        self.show_hands = False
        self.show_face = False
        self.show_pose = False

        # Inicializar mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

        self.mp_face = mp.solutions.face_mesh
        self.face = self.mp_face.FaceMesh()

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)

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

    def is_hand_near_face(self, hand_landmarks, face_landmarks, threshold=0.03):
        for hand_point in hand_landmarks:
            for face_point in face_landmarks:
                distance = np.sqrt((hand_point.x - face_point.x) ** 2 + (hand_point.y - face_point.y) ** 2)
                if distance < threshold:
                    return True
        return False

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

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 2)
        face = np.array([[res.x, res.y] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 2)
        lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 2)
        rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 2)
        return np.concatenate([pose, face, lh, rh])

    def predict_dynamic_action(self):
        # Convertir secuencia a formato de entrada para TensorFlow Lite
        input_data = np.expand_dims(self.sequence, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        predicted_class_index = np.argmax(prediction)
        predicted_label = self.labels_dict_dynamics[predicted_class_index]
        return predicted_label

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
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results_hands = self.hands.process(frame_rgb)
                results_face = self.face.process(frame_rgb)
                results_pose = self.pose.process(frame_rgb)
                results_holistic = self.holistic.process(frame_rgb)

                num_hands = 0
                face_detected = False
                face_data_aux = []
                hand_data_aux = []
                pose_data_aux = []

                if results_hands.multi_hand_landmarks:
                    num_hands = len(results_hands.multi_hand_landmarks)
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        hand_data_aux = []
                        for landmark in hand_landmarks.landmark:
                            xh = landmark.x
                            yh = landmark.y
                            hand_data_aux.append(xh)
                            hand_data_aux.append(yh)

                if results_face.multi_face_landmarks:
                    face_detected = True
                    for face_landmarks in results_face.multi_face_landmarks:
                        for landmark in face_landmarks.landmark:
                            xf = landmark.x
                            yf = landmark.y
                            face_data_aux.append(xf)
                            face_data_aux.append(yf)

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

                predicted_character = 'Traducción...'

                # Procesamiento principal en cada frame
                keypoints = self.extract_keypoints(results_holistic)  # Extraer keypoints del frame actual

                # Verificar si hay landmarks en las manos en el frame actual
                hand_present = results_holistic.left_hand_landmarks or results_holistic.right_hand_landmarks

                # Solo agregamos los keypoints si hay landmarks de las manos presentes
                if hand_present:
                    self.sequence.append(keypoints)

                self.sequence = self.sequence[-30:]

                if num_hands >= 1 and face_detected:
                    if self.is_hand_near_face(results_hands.multi_hand_landmarks[0].landmark, results_face.multi_face_landmarks[0].landmark):

                        if self.is_dynamic_sign_active():
                            if len(self.sequence) == 30 and self.frame_count % 1 == 0:  # Se hace la predicción cada 30 segundos.
                                predicted_character = self.predict_dynamic_action()  # Predecir la acción dinámica
                        else:
                            combination = hand_data_aux + face_data_aux
                            prediction = self.model3.predict([np.asarray(combination)])
                            predicted_character = self.labels_dict3[int(prediction[0])]
                    else:
                        if num_hands == 1:
                            if self.is_dynamic_sign_active():
                                if len(self.sequence) == 30 and self.frame_count % 1 == 0:  # Se hace la predicción cada 30 segundos.
                                    predicted_character = self.predict_dynamic_action()  # Predecir la acción dinámica
                            else:
                                prediction = self.model1.predict([np.asarray(hand_data_aux)])
                                predicted_character = self.labels_dict[int(prediction[0])]
                        else:
                            if self.is_dynamic_sign_active():
                                if len(self.sequence) == 30 and self.frame_count % 1 == 0:  # Se hace la predicción cada 30 segundos.
                                    predicted_character = self.predict_dynamic_action()  # Predecir la acción dinámica
                            else:
                                prediction = self.model2.predict([np.asarray(hand_data_aux)])
                                predicted_character = self.labels_dict2[int(prediction[0])]

                if num_hands == 1 and self.is_elbow_visible(results_pose.pose_landmarks, results_hands.multi_hand_landmarks[0]):
                    combinationByH = hand_data_aux + pose_data_aux
                    prediction = self.model4.predict([np.asarray(combinationByH)])
                    predicted_character = self.labels_dict4[int(prediction[0])]

                self.frame_count += 1
                self.result_captured.emit(predicted_character)

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

                h, w, ch = frame_rgb.shape
                qt_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                self.frame_captured.emit(qt_image)

    def stop(self):
        self.running = False
        self.cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Configurar la ventana principal
        self.setWindowTitle("LENSEGUAtraductor")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background-color: #DBE2EA;")

        # Widget central y layout
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central_widget)

        # Crear una escena y vista para el canvas
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        layout.addWidget(self.view)

        # Desactivar barras de desplazamiento
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Asegurarse de que la vista esté alineada correctamente
        self.view.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # Agregar un rectángulo a la escena
        self.scene.addRect(
            0.0, 0.0, 1280.0, 60.0,
            pen=QPen(Qt.PenStyle.NoPen),
            brush=QColor("#FFFFFF")
        )

        # Encabezado
        self.add_text("LENSEGUAtraductor", 16.0, 8.0, 28, QColor("#000000"))

        # Frame para inferencia
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

    def add_text(self, text, x, y, font_size, color, bold=True):
        text_item = QGraphicsTextItem(text)
        font = QFont("IBM Plex Sans", font_size, QFont.Weight.Bold if bold else QFont.Weight.Normal)
        text_item.setFont(font)
        text_item.setDefaultTextColor(color)
        text_item.setPos(x, y)
        self.scene.addItem(text_item)
        return text_item

    def add_image(self, image_path, x, y):
        pixmap = QPixmap(image_path)
        item = QGraphicsPixmapItem(pixmap)
        item.setOffset(x, y)
        self.scene.addItem(item)

    def create_button(self, image_path, x, y, width, height, callback):
        button = QPushButton()

        #Imagen para cuando no se presiona
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

    def set_button_image(self, button, image_path):
        pixmap = QPixmap(image_path)
        icon = QIcon(pixmap)
        button.setIcon(icon)
        button.setIconSize(pixmap.size())
        button.setFixedSize(pixmap.size())

    def text_button_clicked(self):
        print("Text button clicked")

        if not self.update_text:
            self.set_button_image(self.text_button, self.text_button_images[1])
            self.set_button_image(self.audio_button, self.audio_button_images[0])

            self.animate_button(self.text_button)
            self.update_text = True  # Activar la actualización del texto
            self.update_audio = False

    def audio_button_clicked(self):
        print("Audio button clicked")

        if not self.update_audio:
            self.set_button_image(self.audio_button, self.audio_button_images[1])
            self.set_button_image(self.text_button, self.text_button_images[0])

            self.animate_button(self.audio_button)
            self.update_text = False  # Desactivar la actualización del texto
            self.update_audio = True

    def handle_result_captured(self, translation_text):
        if self.update_text:
            self.update_translation(translation_text)
        if self.update_audio:
            self.play_audio(translation_text)

    def update_translation(self, translation_text):
        self.translation_text_item.setPlainText(translation_text)

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

    def animate_button(self, button):
        animation = self.animations[button]
        animation.setDirection(QPropertyAnimation.Direction.Forward)
        animation.finished.connect(lambda: self.reset_button_geometry(button))
        animation.start()

    def reset_button_geometry(self, button):
        animation = self.animations[button]
        animation.finished.disconnect()
        animation.setDirection(QPropertyAnimation.Direction.Backward)
        animation.start()
        animation.finished.connect(lambda: self.restore_button_position(button))

    def restore_button_position(self, button):
        animation = self.animations[button]
        animation.finished.disconnect()
        original_geometry = animation.startValue()
        button.setGeometry(original_geometry)

    def create_switch(self, x, y, switch_type):
        switch_on = QPixmap("switch_on.png")
        switch_off = QPixmap("switch_off.png")

        button_switch = QPushButton()
        button_switch.setIcon(QIcon(switch_off))
        button_switch.setIconSize(switch_off.size())
        button_switch.setFixedSize(switch_off.size())
        button_switch.setFlat(True)
        button_switch.setStyleSheet("border: none; padding: 0px; margin: 0px; background: transparent;")

        button_switch.switch_state = False  # Añadir atributo de estado
        button_switch.switch_on = switch_on
        button_switch.switch_off = switch_off
        button_switch.clicked.connect(lambda: self.switch(button_switch, switch_type))

        self.switches.append(button_switch)
        self.add_widget_to_scene(button_switch, x, y)

    def add_widget_to_scene(self, widget, x, y):
        proxy = QGraphicsProxyWidget()
        proxy.setWidget(widget)
        proxy.setPos(x, y)
        widget_size = widget.size()
        proxy.resize(widget_size.width(), widget_size.height())
        self.scene.addItem(proxy)

    def switch(self, button_switch, switch_type):
        if button_switch.switch_state:
            button_switch.setIcon(QIcon(button_switch.switch_off))
            button_switch.switch_state = False
        else:
            button_switch.setIcon(QIcon(button_switch.switch_on))
            button_switch.switch_state = True

        self.update_switch_size(button_switch)
        self.toggle_switch(switch_type)

    def update_switch_size(self, button_switch):
        if button_switch.switch_state:
            size = button_switch.switch_on.size()
        else:
            size = button_switch.switch_off.size()

        button_switch.setFixedSize(size)

        for item in self.scene.items():
            if isinstance(item, QGraphicsProxyWidget) and item.widget() == button_switch:
                # Obtener la posición actual del item
                current_pos = item.pos()
                # Ajustar la geometría manteniendo la posición actual
                item.setGeometry(QRectF(0, 0, size.width(), size.height()))
                item.setPos(current_pos)  # Mantener la posición actual
                break

    def toggle_switch(self, switch_type):
        if switch_type == 'hands':
            self.video_thread.show_hands = not self.video_thread.show_hands
        elif switch_type == 'face':
            self.video_thread.show_face = not self.video_thread.show_face
        elif switch_type == 'pose':
            self.video_thread.show_pose = not self.video_thread.show_pose

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

    def closeEvent(self, event):
        self.video_thread.stop()
        self.video_thread.wait()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
