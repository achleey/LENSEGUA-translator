class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Configurar la ventana principal
        self.setWindowTitle("LENSEGUAtraductor")        # Título de la ventana
        self.setGeometry(100, 100, 1280, 720)        # Tamaño y posición de la ventana
        self.setStyleSheet("background-color: #DBE2EA;")        # Color de fondo

        # Widget central y layout para organizar los elementos
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)        # Sin márgenes
        self.setCentralWidget(central_widget)

        # Crear una escena y vista para el canvas
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)        # Vista para mostrar la escena
        layout.addWidget(self.view)

        # Desactivar barras de desplazamiento
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Asegurarse de que la vista esté alineada correctamente
        self.view.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # Agregar un rectángulo a la escena como fondo para la cabecera
        self.scene.addRect(
            0.0, 0.0, 1280.0, 60.0,
            pen=QPen(Qt.PenStyle.NoPen),        # Sin borde
            brush=QColor("#FFFFFF")        # Color de fondo blanco
        )

        # Encabezado de la aplicación
        self.add_text("LENSEGUAtraductor", 16.0, 8.0, 28, QColor("#000000"))

        # Frame para inferencia del video
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

        # Área de traducción la traducción en texto
        self.add_image("image_2.png", 738.0, 313.8)
        self.add_text("Traducción a texto", 879.0, 357.8, 28, QColor("#000000"))
        self.add_image("image_3.png", 863.0, 407.8)
        self.translation_text_item = self.add_text("", 883.0, 424.8, 16, QColor("#1F285B"), bold=False)

        # Opciones de traducción
        self.add_image("image_1.png", 738, 126.8)
        self.add_text("Opciones de traducción", 850.0, 170.8, 28, QColor("#000000"))

        # Crear botones con animaciones para opciones de traducción
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

        # Agregar texto para mostrar los Landmarks
        self.add_text("Landmarks - manos", 80.0, 552.0, 16, QColor("#000000"))
        self.add_text("Landmarks - cara", 80.0, 602.0, 16, QColor("#000000"))
        self.add_text("Landmarks - pose", 80.0, 652.0, 16, QColor("#000000"))

        # Inicializar la lista de switches
        self.switches = []

        # Crear interruptores para controlar la visualización de landmarks
        self.create_switch(275, 557, 'hands')
        self.create_switch(275, 607, 'face')
        self.create_switch(275, 657, 'pose')

        # Inicializar y comenzar el hilo de video
        self.video_thread = VideoThread()
        self.video_thread.frame_captured.connect(self.update_frame)
        self.video_thread.result_captured.connect(self.handle_result_captured)  # Conectar a la función manejadora
        self.video_thread.start()

    def add_text(self, text, x, y, font_size, color, bold=True):
        # Crear un objeto de texto y añadirlo a la escena
        text_item = QGraphicsTextItem(text)
        font = QFont("IBM Plex Sans", font_size, QFont.Weight.Bold if bold else QFont.Weight.Normal)
        text_item.setFont(font)
        text_item.setDefaultTextColor(color)
        text_item.setPos(x, y)
        self.scene.addItem(text_item)
        return text_item

    def add_image(self, image_path, x, y):
        # Cargar una imagen y añadirla a la escena 
        pixmap = QPixmap(image_path)
        item = QGraphicsPixmapItem(pixmap)
        item.setOffset(x, y)
        self.scene.addItem(item)

    def create_button(self, image_path, x, y, width, height, callback):
        # Crear un botón con animación para las opciones de traducción
        button = QPushButton()

        # Establecer la imagen del botón
        pixmap = QPixmap(image_path)
        icon = QIcon(pixmap)
        button.setIcon(icon)
        button.setIconSize(pixmap.size())
        button.setFixedSize(pixmap.size())
        button.setFlat(True)
        button.setStyleSheet("border: none; padding: 0px; margin: 0px;")
        button.clicked.connect(callback)
        button.setGeometry(int(x), int(y), int(width), int(height))

        # Crear la animación del botón
        animation = QPropertyAnimation(button, b"geometry")
        animation.setDuration(100)
        animation.setStartValue(QRect(int(x), int(y), int(width), int(height)))
        animation.setEndValue(QRect(int(x) + 5, int(y) + 5, int(width) - 10, int(height) - 10))

        return button, animation

    def set_button_image(self, button, image_path):
        # Cambiar la imagen de un botón
        pixmap = QPixmap(image_path)
        icon = QIcon(pixmap)
        button.setIcon(icon)
        button.setIconSize(pixmap.size())
        button.setFixedSize(pixmap.size())

    def text_button_clicked(self):
        # Acción cuando se hace click en el botón de texto
        print("Text button clicked")

        if not self.update_text:
            self.set_button_image(self.text_button, self.text_button_images[1])
            self.set_button_image(self.audio_button, self.audio_button_images[0])

            self.animate_button(self.text_button)
            self.update_text = True  # Activar la actualización del texto
            self.update_audio = False

    def audio_button_clicked(self):
        # Acción cuando se hace click en el botón de audio
        print("Audio button clicked")

        if not self.update_audio:
            self.set_button_image(self.audio_button, self.audio_button_images[1])
            self.set_button_image(self.text_button, self.text_button_images[0])

            self.animate_button(self.audio_button)
            self.update_text = False  # Desactivar la actualización del texto
            self.update_audio = True

    def handle_result_captured(self, translation_text):
        # Manejar los resultados de la traducción capturados (texto o audio)
        if self.update_text:
            self.update_translation(translation_text)
        if self.update_audio:
            self.play_audio(translation_text)

    def update_translation(self, translation_text):
        # Actualizar el texto de la traducción en la interfaz
        self.translation_text_item.setPlainText(translation_text)

    def play_audio(self, translation_text):
        # Reproducir audio de la traducción
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
        # Reproducir la animación del botón
        animation = self.animations[button]
        animation.setDirection(QPropertyAnimation.Direction.Forward)
        animation.finished.connect(lambda: self.reset_button_geometry(button))
        animation.start()

    def reset_button_geometry(self, button):
        # Resetear la geometría de los botones después de la animación
        animation = self.animations[button]
        animation.finished.disconnect()
        animation.setDirection(QPropertyAnimation.Direction.Backward)
        animation.start()
        animation.finished.connect(lambda: self.restore_button_position(button))

    def restore_button_position(self, button):
        # Asegurar que el botón se mantenga en su posición al momento de la animación
        animation = self.animations[button]
        animation.finished.disconnect()
        original_geometry = animation.startValue()
        button.setGeometry(original_geometry)

    def create_switch(self, x, y, switch_type):
        # Crear interruptores
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
        # Añadir widget a la ventana
        proxy = QGraphicsProxyWidget()
        proxy.setWidget(widget)
        proxy.setPos(x, y)
        widget_size = widget.size()
        proxy.resize(widget_size.width(), widget_size.height())
        self.scene.addItem(proxy)

    def switch(self, button_switch, switch_type):
        # Controlar imagen de switch según su estado
        if button_switch.switch_state:
            button_switch.setIcon(QIcon(button_switch.switch_off))
            button_switch.switch_state = False
        else:
            button_switch.setIcon(QIcon(button_switch.switch_on))
            button_switch.switch_state = True

        self.update_switch_size(button_switch)
        self.toggle_switch(switch_type)

    def update_switch_size(self, button_switch):
        # Actualizar el tamaño del switch si esta habilitado o deshabilitado
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
        # Habilitar o deshabilitar la visualización de landmarks según el switch activo
        if switch_type == 'hands':
            self.video_thread.show_hands = not self.video_thread.show_hands
        elif switch_type == 'face':
            self.video_thread.show_face = not self.video_thread.show_face
        elif switch_type == 'pose':
            self.video_thread.show_pose = not self.video_thread.show_pose

    def update_frame(self, qt_image):
        # Definir dimensiones para el frame de video en tiempo real
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
        # Terminar evento
        self.video_thread.stop()
        self.video_thread.wait()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
