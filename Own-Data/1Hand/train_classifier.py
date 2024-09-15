import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import randint
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

data_dict = pickle.load(open('./Data1HandP.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

rf = RandomForestClassifier()
model = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=5, cv=5)
model.fit(X_train, Y_train)
y_predict = model.predict(X_test)

accuracy = accuracy_score(Y_test, y_predict)
recall = recall_score(Y_test, y_predict, average='macro')
precision = precision_score(Y_test, y_predict, average='macro')
f1 = f1_score(Y_test, y_predict, average='macro')

print('Accuracy: {:.3f}%'.format(accuracy*100))
print('Recall: {:.3f}%'.format(precision*100))
print('Precision: {:.3f}%'.format(recall*100))
print('F1: {:.3f}%'.format(f1*100))

class_names = ['A', 'B', 'C', 'D', 'E', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'U', 'V', 'W', 'Y']

contingency_table = confusion_matrix(Y_test, y_predict)

plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Verdadera')
plt.show()

f = open('Model1HandP.p', 'wb')
pickle.dump({'model': model},f)
f.close()





