import pickle  # Para cargar datos serializados
import os  # Para manejar rutas de archivos

# Importación de módulos para construir y entrenar modelos, dividir datos y métricas
from keras.src.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Para graficar la matriz de confusión
import matplotlib.pyplot as plt
import seaborn as sns  

import numpy as np  # Para trabajar con matrices y datos numéricos

# Función para crear secuencias de datos de longitud especificada
def create_sequences(data, labels, seq_length):
    sequences = []
    sequence_labels = []
    for i in range(len(data) - seq_length + 1):  # Recorre todos los frames
        sequence = data[i:i + seq_length]  # Agrupa en secuencias
        sequences.append(sequence)
        sequence_labels.append(labels[i + seq_length - 1])  # Usa el label del último frame
    return np.array(sequences), np.array(sequence_labels)

# Cargar datos y etiquetas desde archivo serializado
data_dict = pickle.load(open('./DataHolisticV.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Definir la longitud de la secuencia
seq_length = 30

# Generar datos de secuencia y etiquetas correspondientes
X, Y = create_sequences(data, labels, seq_length)

# Convertir etiquetas a formato one-hot
labels_one_hot = to_categorical(Y, num_classes=3)

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, labels_one_hot, test_size=0.2, random_state=42, shuffle=True, stratify=Y)

# Configurar TensorBoard para visualización
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Configurar Early Stopping para detener entrenamiento cuando no mejora
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=10, verbose=1, mode="min", restore_best_weights=True)

# Definir el modelo de red neuronal secuencial
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1086)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Capa de salida para clasificación de 3 clases

# Configurar el optimizador y compilar el modelo
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Entrenar el modelo
history = model.fit(X_train, Y_train,
                    validation_data=(X_test, Y_test),
                    epochs=2000,
                    callbacks=[tb_callback, early_stopping])

# Realizar predicciones en el conjunto de prueba
y_predict = model.predict(X_test)

# Convertir predicciones y etiquetas de prueba de one-hot a clases
y_predict_classes = np.argmax(y_predict, axis=1)
Y_test_classes = np.argmax(Y_test, axis=1)

# Calcular métricas de evaluación
accuracy = accuracy_score(Y_test_classes, y_predict_classes)
recall = recall_score(Y_test_classes, y_predict_classes, average='macro')
precision = precision_score(Y_test_classes, y_predict_classes, average='macro')
f1 = f1_score(Y_test_classes, y_predict_classes, average='macro')

# Imprimir métricas de evaluación
print('Accuracy: {:.3f}%'.format(accuracy*100))
print('Recall: {:.3f}%'.format(precision*100))
print('Precision: {:.3f}%'.format(recall*100))
print('F1: {:.3f}%'.format(f1*100))

# Definir nombres de clases para matriz de confusión
class_names = ['F', 'J', 'S']

# Calcular y graficar la matriz de confusión
contingency_table = confusion_matrix(Y_test_classes, y_predict_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Verdadera')
plt.show()

# Guardar el modelo
model.export("ModelHolisticV")
