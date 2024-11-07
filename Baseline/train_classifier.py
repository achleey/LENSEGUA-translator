import pickle  # Para guardar y cargar modelos en formato binario

# Importación de módulos de scikit-learn para el modelo y la evaluación
from sklearn.ensemble import RandomForestClassifier    # Clasificador basado en Random Forest
from sklearn.model_selection import train_test_split   # Dividir los datos en conjunto de entrenamiento y prueba
from sklearn.metrics import accuracy_score             # Métrica para evaluar precisión del modelo
import numpy as np                                     # Operaciones con arrays numéricos 

# Cargar los datos y etiquetas desde un archivo en formato binario
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convertir datos y etiquetas a arrays de NumPy
data = np.asarray(data_dict['data'])        
labels = np.asarray(data_dict['labels'])    

# Dividir los datos en conjuntos de entrenamiento (80%) y prueba (20%)
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Inicializar y entrenar el modelo Random Forest
model = RandomForestClassifier()    
model.fit(X_train, Y_train)    

# Predecir etiquetas para el conjunto de prueba
y_predict = model.predict(X_test)   

# Calcular y mostrar precisión del modelo
score = accuracy_score(y_predict, Y_test)       
print('{}% of samples were classified correctly !'.format(score*100))

# Guardar el modelo entrenado en un archivo binario
f = open('model.p', 'wb')   
pickle.dump({'model': model},f)
f.close()
