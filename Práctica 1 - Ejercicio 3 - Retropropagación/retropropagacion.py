import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging


import os 
# Establecer la variable de entorno para desactivar las advertencias de oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
# Desactivar los mensajes de advertencia de TensorFlow
tf.get_logger().setLevel(logging.ERROR)

# Cargar dataset
data = pd.read_csv('concentlite.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1:].values

# Solicitar al usuario la cantidad de capas de la red y la cantidad de neuronas para cada capa
print()
num_layers = int(input("=== [ Ingrese la cantidad de capas de la red neuronal ]: "))
layer_dimensions = []
for i in range(num_layers):
    num_neurons = int(input(f"Ingrese la cantidad de neuronas para la capa {i + 1}: "))
    layer_dimensions.append(num_neurons)
print()   

# Generar 1000 puntos de datos aleatorios
random_points_X = np.random.rand(1000, 2) * 4 - 2  # Rango (-2, 2)
random_points_Y = np.random.randint(0, 2, (1000, 1))  # Etiquetas aleatorias 0 o 1

# Concatenar los puntos de datos aleatorios con los datos originales
X = np.concatenate((X, random_points_X), axis=0)
Y = np.concatenate((Y, random_points_Y), axis=0)

# Crear el modelo utilizando TensorFlow
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(X.shape[1],)))
for num_neurons in layer_dimensions:
    model.add(tf.keras.layers.Dense(num_neurons, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Menú de selección de regla de aprendizaje
print()
print("=== [ Seleccione el tipo de regla de aprendizaje ] ===")
print("1. Retropropagación")
print("2. Binary Crossentropy con Optimizador Adam")
print()

# Solicitar al usuario que seleccione una opción
option = input("Ingrese el número correspondiente a la opción seleccionada: ")
print()

# Entrenamiento con la regla de aprendizaje seleccionada por el usuario
if option == '1':
    # Entrenamiento con retropropagación
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
    history1 = model.fit(X, Y, epochs=500, batch_size=32, verbose=0)

    # Plotear la evolución del costo durante el entrenamiento con retropropagación
    plt.plot(history1.history['loss'])
    plt.xlabel("Iteración")
    plt.ylabel("Costo")
    plt.title("Evolución del costo durante el entrenamiento con retropropagación")
    plt.show()
    
elif option == '2':
    # Entrenamiento con binary_crossentropy y el optimizador 'adam'
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history2 = model.fit(X, Y, epochs=500, batch_size=32, verbose=0)

    # Plotear la evolución del costo durante el entrenamiento con binary_crossentropy y 'adam'
    plt.plot(history2.history['loss'])
    plt.xlabel("Iteración")
    plt.ylabel("Costo")
    plt.title("Evolución del costo durante el entrenamiento con binary_crossentropy y 'adam'")
    plt.show()
else:
    print("Opción no válida.")

# Predicción y visualización del resultado
predictions = model.predict(X)
predictions = np.where(predictions > 0.5, 1, 0)
plt.scatter(X[:, 0], X[:, 1], c=predictions.ravel(), cmap=plt.cm.Spectral)
plt.xlabel('Y')
plt.ylabel('X')
plt.title('Clasificación realizada por la red neuronal')
plt.show()