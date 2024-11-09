import pickle    # Para guardar y cargar datos en formato binario
from sklearn.ensemble import RandomForestClassifier    # Para el modelo Random Forest
from sklearn.model_selection import RandomizedSearchCV, train_test_split    # Para búsqueda de hiperparámetros y división de datos
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score    # Para evaluar el modelo
from scipy.stats import randint    # Para definir distribuciones de hiperparámetros
import matplotlib.pyplot as plt    # Para visualización de gráficos
import seaborn as sns    # Para visualización de datos, en este caso, la matriz de confusión
import numpy as np    # Para operaciones matemáticas y manejo de arrays

# Cargar los datos procesados de landmarks y etiquetas desde el archivo pickle
data_dict = pickle.load(open('./Data1HandP.pickle', 'rb'))

# Convertir los datos y etiquetas en arreglos de numpu para su procesamiento
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Dividir el conjunto de datos en entrenamiento y prueba, con 20% para pruebas
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

# Definición de los rangos de valores de hiperparámetros para la búsqueda aleatoria
param_dist = {'n_estimators': randint(50,500),    # Número de árboles entre 50 y 500
              'max_depth': randint(1,20)}    # Profundidad máxima de los árboles entre 1 y 20      

# Inicializar el modelo Random Forest
rf = RandomForestClassifier()
# Configuración de RandomizedSearchCV para ajustar los hiperparámetros, con 5 iteraciones y validación cruzada 
model = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=5, cv=5)
# Entrenamiento del modelo con los datos de entrenamiento
model.fit(X_train, Y_train)

# Realizar predicciones en el conjunto de prueba
y_predict = model.predict(X_test)

# Calcular las métricas de evaluación del modelo
accuracy = accuracy_score(Y_test, y_predict)
recall = recall_score(Y_test, y_predict, average='macro')
precision = precision_score(Y_test, y_predict, average='macro')
f1 = f1_score(Y_test, y_predict, average='macro')

# Mostrar las métricas de evaluación en formato porcentual
print('Accuracy: {:.3f}%'.format(accuracy*100))
print('Recall: {:.3f}%'.format(precision*100))
print('Precision: {:.3f}%'.format(recall*100))
print('F1: {:.3f}%'.format(f1*100))

# Definir los nombres de las clases para el gráfico de la matriz de confusión
class_names = ['A', 'B', 'C', 'D', 'E', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'U', 'V', 'W', 'Y']

# Calcular la matriz de confusión para comparar las predicciones y las etiquetas reales
contingency_table = confusion_matrix(Y_test, y_predict)

# Graficar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Verdadera')
plt.show()

# Guardar el modelo entrenado en un archivo pickle para uso posterior
f = open('Model1HandP.p', 'wb')
pickle.dump({'model': model},f)
f.close()
