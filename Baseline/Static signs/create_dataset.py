import os                       # Biblioteca para interacción con el sistema operativo.
import pickle                   # Para guardar y cargar datos en archivos binarios
import mediapipe as mp          # Biblioteca para detectar y procesar poses y landmarks
import cv2                      # OpenCV para procesamiento de imágenes
import matplotlib.pyplot as plt     # Biblioteca para visualización de datos

# Configurar módulos de MediaPipe para detección de manos y visualización de landmarks
mp_hands = mp.solutions.hands       
mp_drawing = mp.solutions.drawing_utils     
mp_drawing_styles = mp.solutions.drawing_styles 

# Inicializar el modelo de detección de manos para imágenes estáticas con confianza mínima de 0.3
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) # Modelo de deteccion de manos, estaticas. 1 altamente confiable, 0 no confiable

DATA_DIR = './data'                 # Directorio con imágenes recolectadas
data = []                           # Para almacenar coordenadas de landmarks y etiquetas de clase. 
labels = []             

# Procesar imágenes en cada subdirectorio de DATA_DIR
# Modificado del original para Apple (Chip M1 Pro)
for dir_ in os.listdir(DATA_DIR):  
    dir_path = os.path.join(DATA_DIR, dir_)    # Ruta completa del subdirectorio actual         
    if os.path.isdir(dir_path):                # Ignorar archivos no deseados como .DS_Store en macOS.
        for img_path in os.listdir(dir_path): 
            data_aux = []                      # Lista temporal para coordenadas de landmarks
            img = cv2.imread(os.path.join(dir_path, img_path)) 
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # Convertir de BGR a RGB. 

            # Detectar manos y obtener coordenadas de landmarks
            results = hands.process(img_rgb) 
            if results.multi_hand_landmarks:    # Verificar detección de manos. 
                for hand_landmarks in results.multi_hand_landmarks: 
                    for i in range(len(hand_landmarks.landmark)):    
                        x = hand_landmarks.landmark[i].x    
                        y = hand_landmarks.landmark[i].y   
                        data_aux.append(x)                  # Agrega coordenada x e y al arreglo data_aux
                        data_aux.append(y)                  

                data.append(data_aux)                       # Agregar landmarks a datos
                labels.append(dir_)                         # Agregar etiqueta de clase    

# Guardar datos y etiquetas en un archivo binario para su uso posterior. 
f = open('data.pickle', 'wb')   
pickle.dump({'data': data, 'labels': labels},f) 
f.close()   
