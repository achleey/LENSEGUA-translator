import os                           # Para interactuar con el OS (ej: guardar o abrir carpetas y documentos).

import cv2                          # Procesamiento de imagenes y video.

 DATA_DIR = './data'                 # Esta variable guarda la ruta del directorio donde estan las fotos.
 if not os.path.exists(DATA_DIR):    # Revisa si el directorio existe o no, de no existir.
     os.makedirs(DATA_DIR)           # Lo crea

 number_of_classes = 3               # Se indica el numero de clases que habrán. 3 por el momento.
 dataset_size = 100                  # Se indica el tamaño de muestras para cada clase.

 cap = cv2.VideoCapture(1)           # Se crea un objeto para capturar video de la camara. 1 es el indice segun OS
 for j in range(number_of_classes):  # Se recorre de 0 a 3-1
     if not os.path.exists(os.path.join(DATA_DIR, str(j))):  # Si no hay una carpeta para la clase en el directorio
         os.makedirs(os.path.join(DATA_DIR, str(j)))         # La crea. str(j) para crearla con su nombre en string.

     print('Collecting data for class {}'.format(j))

     done = False
     while True:
         ret, frame = cap.read()     #cap.read lee el objeto cap. Ret indica si se pudo caputrar y frame contiene la img
         cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                     cv2.LINE_AA)    # texto para indicar al usuario que inicie el proceso de captura
         cv2.imshow('frame', frame)  #se muestra el frame al usuario
         if cv2.waitKey(25) == ord('q'):     # al presionar q puede iniciarse el siguiente paso (captura)
             break

     counter = 0
     while counter < dataset_size:   #con counter se controla la cantidad de imagenes capturadas
         ret, frame = cap.read()     #se lee y captura el frame
         cv2.imshow('frame', frame) #se muestra lo que se esta capturando
         cv2.waitKey(25)             # Se esperan 25 milisegundos entre cada frame
         cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame) #se guarda la imagen en el
         #directorio y carpeta correspondiente. {} se reemplaza por el valor del counter
         counter += 1

 cap.release()       # se detiene la captura de video
 cv2.destroyAllWindows()     # se cierran las ventanas de visualizacion de video
