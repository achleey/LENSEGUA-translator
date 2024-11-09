import os        # Librería para interactuar con el sistema de archivos, manejar directorios, y rutas de archivos
import cv2       # Librería OpenCV para captura y procesamiento de imágenes

# Configuración del directorio de datos para almacenar las imágenes
DATA_DIR = './2Hands'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3        # Número de clases a capturar
dataset_size = 500        # Cantidad de imágenes por clase

# Inicializar la captura de vídeo desde la cámara
cap = cv2.VideoCapture(1)

# Repetir el proceso de captura para cada clase
for j in range(number_of_classes):
    # Crear subdirectorio para la clase actual si no existe
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    # Esperar la confirmación del usuario para comenzar la captura
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):        # Iniciar captura al presionar 'q'
            break

    # Capturar y guardar imágenes para la clase actual
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

# Liberar recursos de la cámara y cerrar ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()














