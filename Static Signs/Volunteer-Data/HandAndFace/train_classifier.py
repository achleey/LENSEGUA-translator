import pickle  # Para cargar y guardar archivos de objetos serializados
from sklearn.ensemble import RandomForestClassifier  # Importar el clasificador Random Forest
from sklearn.model_selection import RandomizedSearchCV  # Para búsqueda aleatoria de hiperparámetros
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y prueba
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report  # Métricas de evaluación
from scipy.stats import randint  # Para definir distribuciones aleatorias para los hiperparámetros
import matplotlib.pyplot as plt  # Para visualización de gráficos
import seaborn as sns  # Para mejorar la visualización con gráficos

import numpy as np  # Para manejo de arrays y matrices

# Cargar los datos y etiquetas desde un archivo pickle
data_dict = pickle.load(open('./DataHandAndFaceV.pickle', 'rb'))

# Convertir los datos y etiquetas a arrays de numpy
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

# Definir el espacio de búsqueda de hiperparámetros para Random Forest
param_dist = {'n_estimators': randint(50,500),  # Número de árboles
              'max_depth': randint(1,20)}  # Profundidad máxima de los árboles

# Inicializar el clasificador Random Forest
rf = RandomForestClassifier()

# Configurar la búsqueda aleatoria con validación cruzada
model = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=5, cv=5)

# Ajustar el modelo a los datos de entrenamiento
model.fit(X_train, Y_train)

# Realizar predicciones sobre los datos de prueba
y_predict = model.predict(X_test)

# Calcular las métricas de evaluación
score = accuracy_score(Y_test, y_predict)  # Exactitud
recall = recall_score(Y_test, y_predict, average='macro')  # Recall (sensibilidad)
precision = precision_score(Y_test, y_predict, average='macro')  # Precisión
f1 = f1_score(Y_test, y_predict, average='macro')  # F1-score (balance entre precisión y recall)

# Imprimir las métricas
print('Accuracy: {:.3f}%'.format(score*100))
print('Recall: {:.3f}%'.format(recall*100))
print('Precision: {:.3f}%'.format(precision*100))
print('F1: {:.3f}%'.format(f1*100))

# Definir las clases (en este caso 'G', 'H', 'I', 'T')
class_names = ['G', 'H', 'I', 'T']

# Generar la matriz de confusión para evaluar el rendimiento del modelo
contingency_table = confusion_matrix(Y_test, y_predict)

# Crear un gráfico de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues',  # Usar un mapa de color azul
            xticklabels=class_names,  # Etiquetas para las columnas
            yticklabels=class_names)  # Etiquetas para las filas
plt.title('Matriz de Confusión')  # Título del gráfico
plt.xlabel('Predicción')  # Etiqueta para el eje X
plt.ylabel('Etiqueta Verdadera')  # Etiqueta para el eje Y
plt.show()  # Mostrar el gráfico

# Guardar el modelo entrenado en un archivo pickle
f = open('ModelHandAndFaceV.p', 'wb')
pickle.dump({'model': model}, f)  # Guardar el modelo en el archivo
f.close()  # Cerrar el archivo
