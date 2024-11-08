import pickle  # Para cargar y guardar objetos de Python en archivos
from sklearn.ensemble import RandomForestClassifier  # Importa el clasificador Random Forest
from sklearn.model_selection import RandomizedSearchCV  # Para búsqueda aleatoria de hiperparámetros
from sklearn.model_selection import train_test_split  # Para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report  # Métricas de evaluación
from scipy.stats import randint  # Para generar distribuciones de enteros aleatorios en la búsqueda de hiperparámetros
import matplotlib.pyplot as plt  # Para visualización de gráficos
import seaborn as sns  # Opcional para mejorar la visualización de la matriz de confusión
import numpy as np  # Para manejar arreglos y operaciones numéricas

# Cargar los datos desde un archivo pickle
data_dict = pickle.load(open('./Data2HandsV.pickle', 'rb'))

# Convertir los datos y etiquetas a arreglos numpy
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

# Definir el rango de parámetros para la búsqueda aleatoria (número de árboles y profundidad máxima)
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Inicializar el clasificador Random Forest
rf = RandomForestClassifier()

# Configurar la búsqueda aleatoria de hiperparámetros con validación cruzada
model = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=5, cv=5)

# Ajustar el modelo a los datos de entrenamiento
model.fit(X_train, Y_train)

# Realizar predicciones sobre el conjunto de prueba
y_predict = model.predict(X_test)

# Calcular las métricas de evaluación
score = accuracy_score(Y_test, y_predict)
recall = recall_score(Y_test, y_predict, average='macro')
precision = precision_score(Y_test, y_predict, average='macro')
f1 = f1_score(Y_test, y_predict, average='macro')

# Imprimir las métricas de evaluación
print('Accuracy: {:.3f}%'.format(score*100))
print('Recall: {:.3f}%'.format(recall*100))
print('Precision: {:.3f}%'.format(precision*100))
print('F1: {:.3f}%'.format(f1*100))

# Definir las clases para las etiquetas
class_names = ['Ñ', 'Q', 'X']

# Calcular la matriz de confusión
contingency_table = confusion_matrix(Y_test, y_predict)

# Visualizar la matriz de confusión con un mapa de calor
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Verdadera')
plt.show()

# Guardar el modelo entrenado en un archivo pickle
f = open('Model2HandsV.p', 'wb')
pickle.dump({'model': model},f)
f.close()
