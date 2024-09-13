import os
import pickle                   #sirve para guardar y abrir archivos

import mediapipe as mp          # Para analizar datos
import cv2
import matplotlib.pyplot as plt     # Para visualizar y graficar imagenes

mp_hands = mp.solutions.hands       # Se importa el modulo hands del paquete solutions. Para detectar manos
mp_drawing = mp.solutions.drawing_utils     # Importar drawing_utils de solutions. Para dibujar sobre imgs y videos.
mp_drawing_styles = mp.solutions.drawing_styles #Importar drawing_styles de solutions. Para personalizar apariencia de dibujos.

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) # Modelo de deteccion de manos, estaticas. 1 altamente confiable, 0 no confiable

DATA_DIR = './data'                 # Ruta de directorio de datos a procesar

data = []               # se guardan las caracteristicas extraidas (coordenadas x y y de los landmarks)
labels = []             # categoria (clase a la que pertenece)

# Modificado del original para Apple (M1 Pro)

for dir_ in os.listdir(DATA_DIR):  # Itera sobre los elementos en DATA_DIR y devuelve una lista
    dir_path = os.path.join(DATA_DIR, dir_) # Aca se une la ruta. Ej: data-0, data-1, data-2
    if os.path.isdir(dir_path): # En macOS hay un archivo .DS_Store que no es un directorio, esto devuelve true si se apunta a un directorio.
        for img_path in os.listdir(dir_path): #Se iteran los archivos en dir_path(las clases) pero solo se itera hasta el primer archivo.
            data_aux = []   #array auxiliar para guardar caracteristica
            img = cv2.imread(os.path.join(dir_path, img_path)) # Se lee la imagen en dir_path - img_path
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # se cambia de BGR a RGB para poder usar con matplotlib y mediapipe

            results = hands.process(img_rgb) #.process procesa la imagen img_rgb con el modelo de deteccion hands.
            if results.multi_hand_landmarks:   #verifica que se detecten manos en results. multi_hand... es una lista con puntos clave de las manos detectadas
                for hand_landmarks in results.multi_hand_landmarks: #itera sobre cada conjunto de puntos claves de la lista de cada mano.
                    for i in range(len(hand_landmarks.landmark)):   #itera sobre los conjuntos de puntos clave de la mano actual (largo de conjunto len, indice de conjunto range) y lo guarda en i
                        x = hand_landmarks.landmark[i].x    #Acceder a coordenada x de conjunto de datos i
                        y = hand_landmarks.landmark[i].y    #Acceder a coordenada y de conjunto de datos i
                        data_aux.append(x)                  #Agrega coordenada x al arreglo data_aux
                        data_aux.append(y)                  #Agrega coordenada y al arreglo data_aux

                data.append(data_aux)                       #Agregamos x y y a arreglo data
                labels.append(dir_)         #Agrega el nombre del directorio dir_ (0,1,2) a labels. De esta forma se asocian los puntos clave con las clases.

f = open('data.pickle', 'wb')   # se abre un archivo llamado data.pickle. w significa que se escribira y b que sera en binario
pickle.dump({'data': data, 'labels': labels},f) # en el archivo se guarda los datos y la clase asociada
f.close()   #se cierra el archivo indicando que se ha terminado la escritura
