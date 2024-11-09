import pickle    # Para cargar y guardar datos en formato binario

# Importar módulos para aprendizaje automático y estadísticas
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
from scipy.stats import randint
import matplotlib.pyplot as plt
import seaborn as sns  # Opcional para mejorar la visualización de la matriz de confusión

import numpy as np    # Para manipulación de arreglos numéricos

# Cargar datos y etiquetas desde archivo pickle
data_dict = pickle.load(open('./Data2HandsP.pickle', 'rb'))
data = np.asarray(data_dict['data'])    # Convertir datos a formato numpy para fácil mamipulación
labels = np.asarray(data_dict['labels'])    # Convertir etiquetas a formato numpy

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

# Definir rango de parámetros para el modelo
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Configurar y ajustar el modelo Random Forest con búsqueda aleatoria
rf = RandomForestClassifier()
model = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=5, cv=5)
model.fit(X_train, Y_train)

# Realizar predicciones y calcular métricas
y_predict = model.predict(X_test)
score = accuracy_score(Y_test, y_predict)
recall = recall_score(Y_test, y_predict, average='macro')
precision = precision_score(Y_test, y_predict, average='macro')
f1 = f1_score(Y_test, y_predict, average='macro')

# Imprimir resultados
print('Accuracy: {:.3f}%'.format(score*100))
print('Recall: {:.3f}%'.format(recall*100))
print('Precision: {:.3f}%'.format(precision*100))
print('F1: {:.3f}%'.format(f1*100))

# Mostrar matriz de confusión
class_names = ['Ñ', 'Q', 'X']
contingency_table = confusion_matrix(Y_test, y_predict)
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Verdadera')
plt.show()

# Guardar el modelo entrenado
f = open('Model2HandsP.p', 'wb')
pickle.dump({'model': model},f)
f.close()





