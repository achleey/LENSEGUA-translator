import os  # Para manejar directorios y rutas de archivos
import cv2  # Librería para capturar y procesar video

# Definir el directorio donde se guardarán las imágenes
DATA_DIR = './HandAndFace'
if not os.path.exists(DATA_DIR):  # Crear el directorio si no existe
    os.makedirs(DATA_DIR)

# Número de clases y tamaño del dataset por clase
number_of_classes = 4
dataset_size = 500

# Iniciar la captura de video (usar cámara con índice 1)
cap = cv2.VideoCapture(1)
for j in range(number_of_classes):
    # Crear directorio para cada clase si no existe
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    # Esperar hasta que el usuario esté listo para capturar imágenes
    while True:
        ret, frame = cap.read()  # Leer el frame de la cámara
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Mostrar el frame en pantalla
        if cv2.waitKey(25) == ord('q'):  # Salir del bucle si se presiona 'q'
            break

    # Capturar las imágenes para el dataset
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()  # Leer el frame de la cámara
        cv2.imshow('frame', frame)  # Mostrar el frame en pantalla
        cv2.waitKey(25)
        # Guardar el frame en el directorio correspondiente
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

# Liberar la captura de video y cerrar todas las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()
