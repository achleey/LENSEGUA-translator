import pickle  # Para cargar y guardar objetos serializados
from sklearn.ensemble import RandomForestClassifier  # Importa el clasificador RandomForest
from sklearn.model_selection import RandomizedSearchCV  # Para búsqueda aleatoria de hiperparámetros
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y prueba
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report  # Métricas de evaluación
from scipy.stats import randint  # Para definir distribuciones aleatorias de enteros
import matplotlib.pyplot as plt  # Para la visualización de resultados
import seaborn as sns  # Opcional para mejorar la visualización con mapas de calor
import numpy as np  # Para manipulación de arrays

# Cargar los datos serializados desde el archivo pickle
data_dict = pickle.load(open('./DataHandAndBodyV.pickle', 'rb'))

# Extraer los datos y las etiquetas de los datos cargados
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividir los datos en conjuntos de entrenamiento y prueba (80% - 20%)
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

# Definir el rango de parámetros para la búsqueda aleatoria
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Crear el modelo RandomForest y configurarlo para la búsqueda aleatoria
rf = RandomForestClassifier()
model = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=5, cv=5)

# Ajustar el modelo con los datos de entrenamiento
model.fit(X_train, Y_train)

# Hacer predicciones sobre el conjunto de prueba
y_predict = model.predict(X_test)

# Evaluar el modelo usando métricas de clasificación
score = accuracy_score(Y_test, y_predict)
recall = recall_score(Y_test, y_predict, average='macro')
precision = precision_score(Y_test, y_predict, average='macro')
f1 = f1_score(Y_test, y_predict, average='macro')

# Imprimir las métricas de evaluación
print('Accuracy: {:.3f}%'.format(score*100))
print('Recall: {:.3f}%'.format(recall*100))
print('Precision: {:.3f}%'.format(precision*100))
print('F1: {:.3f}%'.format(f1*100))

# Definir los nombres de las clases para la matriz de confusión
class_names = ['Z']

# Calcular la matriz de confusión
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
f = open('ModelHandAndBodyV.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
