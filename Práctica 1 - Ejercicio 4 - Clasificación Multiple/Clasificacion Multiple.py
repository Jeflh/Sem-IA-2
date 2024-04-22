import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut, LeavePOut
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Cargar los datos
data = pd.read_csv("irisbin.csv")

# Separar características (X) y etiquetas (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalizar las características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir los datos en conjunto de entrenamiento y prueba
random_state = int(time.time() % 1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state)

# Construir el clasificador MLP
mlp_classifier = MLPClassifier(hidden_layer_sizes=(20, 20, 20), activation='relu', max_iter=1000)

# Entrenar el clasificador MLP
mlp_classifier.fit(X_train, y_train)

# Evaluar el rendimiento en el conjunto de prueba
accuracy = accuracy_score(y_test, mlp_classifier.predict(X_test))
print("Accuracy en conjunto de prueba: {:.2f}%".format(accuracy * 100))

# Validación cruzada: leave-one-out
loo = LeaveOneOut()
loo_scores = []
for train_index, test_index in loo.split(X):
    X_train_loo, X_test_loo = X[train_index], X[test_index]
    y_train_loo, y_test_loo = y[train_index], y[test_index]
    mlp_classifier_loo = MLPClassifier(hidden_layer_sizes=(20, 20, 20), activation='relu', max_iter=1000)
    mlp_classifier_loo.fit(X_train_loo, y_train_loo)
    loo_scores.append(accuracy_score(y_test_loo, mlp_classifier_loo.predict(X_test_loo)))

loo_mean_accuracy = np.mean(loo_scores)
loo_std_accuracy = np.std(loo_scores)
print("Leave-One-Out - Promedio de precisión: {:.2f}%".format(loo_mean_accuracy * 100))
print("Leave-One-Out - Desviación estándar de precisión:  {:.2f}%".format(loo_std_accuracy * 100))

# Validación cruzada: leave-p-out
p_out = 1  # Modificar el valor de p según sea necesario
lpout = LeavePOut(p=p_out)
lpout_scores = []
for train_index, test_index in lpout.split(X):
    X_train_lpout, X_test_lpout = X[train_index], X[test_index]
    y_train_lpout, y_test_lpout = y[train_index], y[test_index]
    mlp_classifier_lpout = MLPClassifier(hidden_layer_sizes=(20, 20, 20), activation='relu', max_iter=1000)
    mlp_classifier_lpout.fit(X_train_lpout, y_train_lpout)
    lpout_scores.append(accuracy_score(y_test_lpout, mlp_classifier_lpout.predict(X_test_lpout)))

lpout_mean_accuracy = np.mean(lpout_scores)
lpout_std_accuracy = np.std(lpout_scores)
print(f"Leave-{p_out}-Out - Promedio de precisión: {lpout_mean_accuracy*100:.2f}%")
print(f"Leave-{p_out}-Out - Desviación estándar de precisión: {lpout_std_accuracy*100:.2f}%")
