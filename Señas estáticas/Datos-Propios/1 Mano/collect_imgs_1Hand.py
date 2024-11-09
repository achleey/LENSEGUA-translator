import os        # Para manejo de directorios
import cv2       # Para captura de vídeo y manipulación de imágenes

# Directorio principal para almanecar el dataset de imágenes
DATA_DIR = './1Hand'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)    # Crear directorio si no existe

# Número de clases a capturar y tamaño de dataset por clase
number_of_classes = 16
dataset_size = 500

# Iniciar captura de video desde la cámara (index 1)
cap = cv2.VideoCapture(1)

# Crear subdirectorios para cada clase y capturar imágenes
for j in range(number_of_classes):
    # Crear un subdirectorio para cada clase si no existe 
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    # Espera de confirmación antes de comenzar la captura 
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)    # Mensaje de inicio
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):    # Iniciar captura cuando se presiona 'q'
            break

    # Captura y guardar 'dataset_size' imágenes para la clase actual 
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

# Liberar la captura de vídeo y cerrar ventanas de OpenCV. 
cap.release()
cv2.destroyAllWindows()
