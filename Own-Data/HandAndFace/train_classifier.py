import pickle  # Para cargar y guardar datos en formato binario

from sklearn.ensemble import RandomForestClassifier  # Clasificador Random Forest
from sklearn.model_selection import RandomizedSearchCV, train_test_split  # Búsqueda de hiperparámetros y división de datos
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix  # Métricas de evaluación
from scipy.stats import randint  # Distribución aleatoria de enteros para la búsqueda de hiperparámetros
import matplotlib.pyplot as plt  # Para graficar
import seaborn as sns  # Opcional, mejora visualización de gráficos
import numpy as np  # Para manejo de arreglos

# Cargar datos y etiquetas del archivo pickle
data_dict = pickle.load(open('./DataHandAndFaceP.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividir datos en conjuntos de entrenamiento y prueba (80/20) con estratificación
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

# Definir distribuciones aleatorias para la búsqueda de hiperparámetros
param_dist = {'n_estimators': randint(50,500),  # Número de árboles
              'max_depth': randint(1,20)}       # Profundidad máxima de los árboles

# Configurar modelo Random Forest con búsqueda aleatoria de hiperparámetros
rf = RandomForestClassifier()
model = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)  # 5 iteraciones de búsqueda, validación cruzada 5 veces
model.fit(X_train, Y_train)  # Entrenar modelo
y_predict = model.predict(X_test)  # Predecir sobre datos de prueba

# Calcular métricas de rendimiento
score = accuracy_score(Y_test, y_predict)
recall = recall_score(Y_test, y_predict, average='macro')
precision = precision_score(Y_test, y_predict, average='macro')
f1 = f1_score(Y_test, y_predict, average='macro')

# Imprimir resultados de métricas
print('Accuracy: {:.3f}%'.format(score*100))
print('Recall: {:.3f}%'.format(recall*100))
print('Precision: {:.3f}%'.format(precision*100))
print('F1: {:.3f}%'.format(f1*100))

# Definir nombres de clases para la matriz de confusión
class_names = ['G', 'H', 'I', 'T']

# Calcular y graficar matriz de confusión
contingency_table = confusion_matrix(Y_test, y_predict)
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Verdadera')
plt.show()

# Guardar modelo entrenado en un archivo pickle
f = open('ModelHandAndFaceP.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
