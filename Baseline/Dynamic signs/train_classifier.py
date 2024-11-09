import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import Adam

# Configuración de rutas y variables globales
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['F', 'J', 'S'])
sequence_length = 30

# Mapeo de etiquetas
label_map = {label:num for num, label in enumerate(actions)}

# Preparación de datos
sequences, labels = [], []
for action in actions:
    # Filtrar archivos no deseados (e.g., .DS_Store)
    action_sequences = [seq for seq in os.listdir(os.path.join(DATA_PATH, action)) if seq.isdigit()]
    for sequence in np.array(action_sequences).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Convertir a arrays y categorías
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Definición del modelo LSTM
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=10, verbose=1, mode="min", restore_best_weights=True)

model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, X.shape[2])),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
])

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Entrenamiento del modelo
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=2000,
                    callbacks=[tb_callback, early_stopping])

# Evaluación del modelo
model.summary()
model.save('action.h5')
model.export("saved_model_action")

# Cargar pesos guardados
model.load_weights('action.h5')

# Predicciones y métricas de evaluación
y_predict = model.predict(X_test)
y_predict_classes = np.argmax(y_predict, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_classes, y_predict_classes)
recall = recall_score(y_test_classes, y_predict_classes, average='macro')
precision = precision_score(y_test_classes, y_predict_classes, average='macro')
f1 = f1_score(y_test_classes, y_predict_classes, average='macro')

print(f'Accuracy: {accuracy*100:.3f}%')
print(f'Recall: {recall*100:.3f}%')
print(f'Precision: {precision*100:.3f}%')
print(f'F1 Score: {f1*100:.3f}%')

# Matriz de Confusión
class_names = ['F', 'J', 'S']
conf_matrix = confusion_matrix(y_test_classes, y_predict_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Verdadera')
plt.show()
