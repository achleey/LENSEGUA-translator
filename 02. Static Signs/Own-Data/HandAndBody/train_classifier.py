import pickle  # Para cargar y guardar modelos y datos

from sklearn.ensemble import RandomForestClassifier  # Clasificador de bosque aleatorio
from sklearn.model_selection import RandomizedSearchCV  # Para la búsqueda de hiperparámetros
from sklearn.model_selection import train_test_split  # División de datos en conjuntos de entrenamiento y prueba
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report  # Métricas de evaluación
from scipy.stats import randint  # Generador de distribuciones para hiperparámetros aleatorios
import matplotlib.pyplot as plt  # Para visualización de datos
import seaborn as sns  # Para mejorar la visualización de gráficos

import numpy as np  # Para operaciones con matrices y arreglos

# Cargar los datos y etiquetas del archivo pickle
data_dict = pickle.load(open('./DataHandAndBodyP.pickle', 'rb'))

# Convertir los datos y etiquetas a arreglos de NumPy
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

# Configuración de hiperparámetros para el modelo RandomForest
param_dist = {'n_estimators': randint(50,500),  # Número de árboles en el bosque
              'max_depth': randint(1,20)}  # Profundidad máxima de los árboles

# Inicializar el clasificador RandomForest con búsqueda aleatoria de hiperparámetros
rf = RandomForestClassifier()
model = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=5, cv=5)
model.fit(X_train, Y_train)  # Entrenar el modelo con los datos de entrenamiento
y_predict = model.predict(X_test)  # Predecir con los datos de prueba

# Calcular las métricas de evaluación
score = accuracy_score(Y_test, y_predict)
recall = recall_score(Y_test, y_predict, average='macro')
precision = precision_score(Y_test, y_predict, average='macro')
f1 = f1_score(Y_test, y_predict, average='macro')

# Imprimir métricas de evaluación
print('Accuracy: {:.3f}%'.format(score*100))
print('Recall: {:.3f}%'.format(recall*100))
print('Precision: {:.3f}%'.format(precision*100))
print('F1: {:.3f}%'.format(f1*100))

# Definir nombres de clases para la matriz de confusión
class_names = ['Z']

# Calcular e imprimir la matriz de confusión
contingency_table = confusion_matrix(Y_test, y_predict)

# Mostrar la matriz de confusión como un mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Verdadera')
plt.show()

# Guardar el modelo entrenado en un archivo pickle
f = open('ModelHandAndBodyP.p', 'wb')
pickle.dump({'model': model},f)
f.close()
