import os                           # Biblioteca para interacción con el sistema operativo
import cv2                          # OpenCV para el procesamiento de imágenes y vídeo en tiempo real

# Configuración del directorio de datos y parámetros de captura
DATA_DIR = './data'                 # Directorio para almacenar las imágenes capturadas.
if not os.path.exists(DATA_DIR):    # Crea el directorio si no existe.
    os.makedirs(DATA_DIR)           

number_of_classes = 3               # Número de clases a capturar
dataset_size = 100                  # Número de imágenes a capturar por clase.

# Inicializar captura de vídeo desde la cámara
cap = cv2.VideoCapture(1)           # Cámara de índice 1 (ajustar si es necesario)

# Bucle para capturar datos de cada clase
for j in range(number_of_classes):  
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):  # Subdirectorio para la clase actual
        os.makedirs(os.path.join(DATA_DIR, str(j)))         # Crear subdirectorio si no existe

    print('Collecting data for class {}'.format(j))         # Mensaje de inicio de captura de clase

    # Instrucciones al usuario antes de iniciar captura
    while True:
        ret, frame = cap.read()    
        frame = cv2.flip(frame, 1)    # Voltea el frame horizontalmente
        cv2.putText(frame, 'Ready? Press "Q" !', (100, 90), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2,
                    cv2.LINE_4)    
        cv2.imshow('Data Capture', frame)  
        if cv2.waitKey(25) == ord('q'):  # Iniciar captura cuando se presione 'q'
            break

    # Captura y guarda imágenes de la clase actual
    counter = 0
    while counter < dataset_size:   
        ret, frame = cap.read()     
        cv2.imshow('frame', frame)  
        cv2.waitKey(25)             
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)  # Guarda la imagen en el directorio de la clase
        counter += 1

# Liberar recursos
cap.release()                        # Libera la cámara
cv2.destroyAllWindows()              # Cierra todas las ventanas de OpenCV
