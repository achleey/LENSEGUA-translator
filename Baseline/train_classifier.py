import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb')) # Se carga el archivo data.pickle. Se indica r para leer y b en binario

data = np.asarray(data_dict['data'])        # Se accede a data del archivo y lo convierte en una lista
labels = np.asarray(data_dict['labels'])    # Se accede a las clases del archivo y lo convierte en una lista.

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
#x y y train tienen los datos para entrenar, x y y test tienen los datos para probar. la funcion de arriba separa el conjunto de datos y se le indico que el test tenga un 20% de tamaño.
#shuffle significa que se randomizan los datos seleccionados. Stratify nos permite usar la misma proporcion de clases para la separacion de nuestros datos.

model = RandomForestClassifier()    # el modelo a usar es un random forest classifier
model.fit(X_train, Y_train)         # se pasan los datos de entrenamiento
y_predict = model.predict(X_test)   # se predice usando los datos de prueba

score = accuracy_score(y_predict, Y_test)       # se mide la precisión del modelo
print('{}% of samples were classified correctly !'.format(score*100))

f = open('model.p', 'wb')   # Guardando el modelo
pickle.dump({'model': model},f)
f.close()
