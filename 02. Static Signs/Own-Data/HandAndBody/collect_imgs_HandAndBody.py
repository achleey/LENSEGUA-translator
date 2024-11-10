import os  # Para trabajar con rutas de archivos y directorios
import cv2  # OpenCV para capturar y procesar imágenes

# Directorio donde se almacenarán las imágenes capturadas
DATA_DIR = './HandAndBody'
if not os.path.exists(DATA_DIR):  
    os.makedirs(DATA_DIR)  # Crea el directorio si no existe

# Configuración de la captura de datos
number_of_classes = 1  # Número de clases de datos a capturar
dataset_size = 500  # Cantidad de imágenes por clase

# Inicializar la captura de video
cap = cv2.VideoCapture(1)  # Usa la cámara secundaria (1)

# Capturar datos para cada clase
for j in range(number_of_classes):
    # Crear una carpeta para cada clase en caso de que no exista
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))  # Informar la clase actual

    # Mostrar mensaje de preparación
    while True:
        ret, frame = cap.read()  # Capturar un cuadro de video
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)  # Mostrar texto en la pantalla
        cv2.imshow('frame', frame)  # Mostrar el cuadro
        if cv2.waitKey(25) == ord('q'):  # Esperar a que el usuario presione "Q" para empezar
            break

    # Capturar y guardar imágenes hasta alcanzar el tamaño del dataset
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()  # Capturar un cuadro de video
        cv2.imshow('frame', frame)  # Mostrar el cuadro
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)  # Guardar imagen en disco
        counter += 1  # Incrementar contador de imágenes capturadas

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
