import os  # Importa el módulo para interactuar con el sistema de archivos
import cv2  # Importa OpenCV para procesamiento de imágenes y video
import time  # Importa módulo para gestionar el tiempo (por ejemplo, pausas)

DATA_DIR = './Videos'  # Define la ruta base para guardar los videos grabados

# Si no existe el directorio para almacenar los videos, lo crea
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3   # Número de clases a grabar (en este caso, 3 clases)
no_sequences = 168      # Número de secuencias a grabar por cada clase (30 videos)
sequence_length = 30    # Longitud de cada secuencia (30 frames por secuencia)

cap = cv2.VideoCapture(1)  # Abre la cámara (1 es el índice de la cámara)

# Recorre cada clase para almacenar los videos
for j in range(number_of_classes):

    # Crea un directorio para cada clase si no existe
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))  # Imprime mensaje indicando que está recolectando datos para la clase j

    # Recorre cada secuencia dentro de la clase
    for sequence in range(no_sequences):
        # Crea un directorio para cada secuencia dentro de la clase
        sequence_dir = os.path.join(class_dir, str(sequence))
        if not os.path.exists(sequence_dir):
            os.makedirs(sequence_dir)

        print('  Collecting sequence {} of class {}'.format(sequence, j))  # Imprime mensaje indicando que está recolectando la secuencia de la clase

        # Muestra un mensaje para indicar que el sistema está listo para capturar
        while True:
            ret, frame = cap.read()  # Captura un frame de la cámara
            frame = cv2.flip(frame, 1)  # Voltea el frame horizontalmente
            cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)  # Muestra un mensaje en el frame
            cv2.imshow('Video Capture', frame)  # Muestra el frame en una ventana
            if cv2.waitKey(25) == ord('q'):  # Espera a que el usuario presione 'q' para comenzar la captura
                break

        counter = 0  # Contador para los frames dentro de la secuencia
        while counter < sequence_length:  # Captura 'sequence_length' frames para la secuencia
            ret, frame = cap.read()  # Captura el siguiente frame
            frame = cv2.flip(frame, 1)  # Aplica el flip horizontal al frame
            # Muestra el mensaje de qué secuencia y clase se está recolectando
            cv2.putText(frame, 'Collecting sequence {} of class {}'.format(sequence, j), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Video Capture', frame)  # Muestra el frame actual
            cv2.waitKey(25)  # Espera 25 ms entre frames
            # Guarda el frame capturado como una imagen JPG dentro del directorio de la secuencia
            cv2.imwrite(os.path.join(class_dir, str(sequence), '{}.jpg'.format(counter)), frame)
            counter += 1  # Incrementa el contador de frames

        time.sleep(1)  # Pausa de 1 segundo después de grabar cada secuencia

cap.release()  # Libera el objeto de captura de video
cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV abiertas
