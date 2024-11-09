import pickle  # Para cargar y guardar objetos en formato pickle
from sklearn.ensemble import RandomForestClassifier  # Clasificador de Bosques Aleatorios
from sklearn.model_selection import RandomizedSearchCV  # Búsqueda aleatoria para encontrar mejores hiperparámetros
from sklearn.model_selection import train_test_split  # Para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report  # Métricas de evaluación
from scipy.stats import randint  # Para generar distribuciones aleatorias para los parámetros
import matplotlib.pyplot as plt  # Para la visualización de gráficos
import seaborn as sns  # Opcional para mejorar la visualización de gráficos

import numpy as np  # Para manipulación de datos numéricos

# Cargar los datos desde un archivo pickle
data_dict = pickle.load(open('./Data1HandV.pickle', 'rb'))

# Convertir los datos y etiquetas a arrays de numpy
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

# Definir el rango de hiperparámetros para la búsqueda aleatoria
param_dist = {'n_estimators': randint(50,500),  # Número de estimadores (árboles)
              'max_depth': randint(1,20)}  # Profundidad máxima de los árboles

# Inicializar el clasificador RandomForest
rf = RandomForestClassifier()

# Inicializar RandomizedSearchCV para encontrar los mejores parámetros
model = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=5, cv=5)

# Entrenar el modelo con el conjunto de entrenamiento
model.fit(X_train, Y_train)

# Predecir las etiquetas en el conjunto de prueba
y_predict = model.predict(X_test)

# Evaluar las métricas de rendimiento
score = accuracy_score(Y_test, y_predict)  # Exactitud
recall = recall_score(Y_test, y_predict, average='macro')  # Recall (sensibilidad) para todas las clases
precision = precision_score(Y_test, y_predict, average='macro')  # Precisión para todas las clases
f1 = f1_score(Y_test, y_predict, average='macro')  # F1-score para todas las clases

# Imprimir las métricas de evaluación
print('Accuracy: {:.3f}%'.format(score*100))
print('Recall: {:.3f}%'.format(recall*100))
print('Precision: {:.3f}%'.format(precision*100))
print('F1: {:.3f}%'.format(f1*100))

# Definir los nombres de las clases (etiquetas)
class_names = ['A', 'B', 'C', 'D', 'E', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'U', 'V', 'W', 'Y']

# Calcular la matriz de confusión
contingency_table = confusion_matrix(Y_test, y_predict)

# Visualizar la matriz de confusión usando un mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues',  # Mostrar la matriz de confusión con anotaciones
            xticklabels=class_names,  # Nombres de las clases en el eje X
            yticklabels=class_names)  # Nombres de las clases en el eje Y
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')  # Etiqueta para el eje X
plt.ylabel('Etiqueta Verdadera')  # Etiqueta para el eje Y
plt.show()

# Guardar el modelo entrenado en un archivo pickle
f = open('Model1HandV.p', 'wb')
pickle.dump({'model': model}, f)  # Guardar el modelo en el archivo
f.close()
