import os  # Importar el módulo os para manejar operaciones de sistema de archivos, como crear rutas
import numpy as np  # Importar NumPy para trabajar con arrays y operaciones matemáticas
from sklearn.model_selection import train_test_split  # Importar la función para dividir los datos en entrenamiento y prueba
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix  # Importar métricas de evaluación para clasificación
import seaborn as sns  # Importar seaborn para visualización avanzada, en particular para la matriz de confusión
import matplotlib.pyplot as plt  # Importar matplotlib para graficar datos
from tensorflow.keras.utils import to_categorical  # Importar para convertir etiquetas en formato one-hot (categoría)
from tensorflow.keras.models import Sequential  # Importar para construir modelos secuenciales de Keras
from tensorflow.keras.layers import LSTM, Dense  # Importar capas LSTM y Dense para el modelo de red neuronal
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping  # Importar callbacks para visualización y detención temprana
from keras.optimizers import Adam  # Importar el optimizador Adam para la red neuronal

# Configuración de rutas y variables globales
DATA_PATH = os.path.join('MP_Data')  # Definir la ruta al directorio que contiene los datos
actions = np.array(['F', 'J', 'S'])  # Definir las acciones a predecir (en este caso las letras 'F', 'J' y 'S')
sequence_length = 30  # Longitud de la secuencia de frames que se usará para la predicción (30 frames)

# Mapeo de etiquetas
label_map = {label:num for num, label in enumerate(actions)}  # Crear un diccionario que asocia cada acción con un número (etiqueta)

# Preparación de datos
sequences, labels = [], []  # Inicializar listas vacías para las secuencias de datos y sus etiquetas
for action in actions:  # Iterar sobre cada acción
    # Filtrar archivos no deseados (e.g., .DS_Store) y asegurarse de que el archivo de la secuencia es un número
    action_sequences = [seq for seq in os.listdir(os.path.join(DATA_PATH, action)) if seq.isdigit()]
    for sequence in np.array(action_sequences).astype(int):  # Convertir las secuencias a números enteros y procesarlas
        window = []  # Crear una lista vacía para almacenar los frames de una secuencia
        for frame_num in range(sequence_length):  # Iterar sobre cada frame de la secuencia (hasta la longitud definida)
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))  # Cargar los datos del frame desde el archivo .npy
            window.append(res)  # Añadir el frame cargado a la ventana de secuencia
        sequences.append(window)  # Añadir la ventana completa a la lista de secuencias
        labels.append(label_map[action])  # Añadir la etiqueta correspondiente a la acción a la lista de etiquetas

# Convertir a arrays y categorías
X = np.array(sequences)  # Convertir las secuencias en un array NumPy
y = to_categorical(labels).astype(int)  # Convertir las etiquetas en formato one-hot (categoría) y asegurarse de que sean enteros

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)  # Dividir los datos en entrenamiento (95%) y prueba (5%)

# Definición del modelo LSTM
log_dir = os.path.join('Logs')  # Definir la ruta para guardar los registros de TensorBoard
tb_callback = TensorBoard(log_dir=log_dir)  # Inicializar el callback de TensorBoard para visualización del entrenamiento
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=10, verbose=1, mode="min", restore_best_weights=True)  # Inicializar el callback de detención temprana

# Construir el modelo LSTM secuencial
model = Sequential([  # Crear un modelo secuencial
    LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, X.shape[2])),  # Primera capa LSTM con 64 unidades, activación ReLU y forma de entrada
    LSTM(128, return_sequences=True, activation='relu'),  # Segunda capa LSTM con 128 unidades y activación ReLU
    LSTM(64, return_sequences=False, activation='relu'),  # Tercera capa LSTM con 64 unidades, sin devolver secuencias
    Dense(64, activation='relu'),  # Capa densa con 64 unidades y activación ReLU
    Dense(32, activation='relu'),  # Capa densa con 32 unidades y activación ReLU
    Dense(actions.shape[0], activation='softmax')  # Capa de salida con tantas neuronas como acciones, activación softmax para clasificación multiclase
])

optimizer = Adam(learning_rate=0.0001)  # Usar el optimizador Adam con una tasa de aprendizaje de 0.0001
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])  # Compilar el modelo con pérdida 'categorical_crossentropy' y métrica de precisión categórica

# Entrenamiento del modelo
history = model.fit(X_train, y_train,  # Entrenar el modelo con los datos de entrenamiento
                    validation_data=(X_test, y_test),  # Validar el modelo con los datos de prueba
                    epochs=2000,  # Número de épocas para entrenar (máximo de 2000)
                    callbacks=[tb_callback, early_stopping])  # Usar callbacks para TensorBoard y EarlyStopping

# Evaluación del modelo
model.summary()  # Mostrar un resumen de la arquitectura del modelo
model.save('action.h5')  # Guardar el modelo entrenado en un archivo 'action.h5'
model.export("saved_model_action")  # Exportar el modelo en un formato compatible con TensorFlow

# Cargar pesos guardados
model.load_weights('action.h5')  # Cargar los pesos previamente guardados del modelo

# Predicciones y métricas de evaluación
y_predict = model.predict(X_test)  # Hacer predicciones con los datos de prueba
y_predict_classes = np.argmax(y_predict, axis=1)  # Obtener las clases predichas (índice con mayor probabilidad)
y_test_classes = np.argmax(y_test, axis=1)  # Obtener las clases verdaderas (índice con mayor probabilidad)

# Calcular métricas de rendimiento
accuracy = accuracy_score(y_test_classes, y_predict_classes)  # Calcular la precisión
recall = recall_score(y_test_classes, y_predict_classes, average='macro')  # Calcular el recall (sensibilidad)
precision = precision_score(y_test_classes, y_predict_classes, average='macro')  # Calcular la precisión
f1 = f1_score(y_test_classes, y_predict_classes, average='macro')  # Calcular la puntuación F1 (balance entre precisión y recall)

# Imprimir las métricas
print(f'Accuracy: {accuracy*100:.3f}%')  # Imprimir precisión en porcentaje
print(f'Recall: {recall*100:.3f}%')  # Imprimir recall en porcentaje
print(f'Precision: {precision*100:.3f}%')  # Imprimir precisión en porcentaje
print(f'F1 Score: {f1*100:.3f}%')  # Imprimir puntuación F1 en porcentaje

# Matriz de Confusión
class_names = ['F', 'J', 'S']  # Definir los nombres de las clases
conf_matrix = confusion_matrix(y_test_classes, y_predict_classes)  # Calcular la matriz de confusión

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))  # Configurar el tamaño de la figura
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',  # Dibujar la matriz de confusión con anotaciones y color azul
            xticklabels=class_names, yticklabels=class_names)  # Etiquetas de las clases en los ejes
plt.title('Matriz de Confusión')  # Título del gráfico
plt.xlabel('Predicción')  # Etiqueta del eje X
plt.ylabel('Etiqueta Verdadera')  # Etiqueta del eje Y
plt.show()  # Mostrar el gráfico
