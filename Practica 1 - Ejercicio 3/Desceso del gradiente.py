import numpy as np
import matplotlib.pyplot as plt

# Definir la función a optimizar
def f(x_1, x_2):
    return 10 - np.exp(-(x_1**2 + 3*x_2**2))

# Calcular el gradiente de la función en un punto dado
def gradient(x_1, x_2):
    df_dx1 = 2*x_1 * np.exp(-(x_1**2 + 3*x_2**2))
    df_dx2 = 6*x_2 * np.exp(-(x_1**2 + 3*x_2**2))
    return np.array([df_dx1, df_dx2])

# Algoritmo de descenso del gradiente
def gradient_descent(lr, iterations):
    # Valores iniciales aleatorios en el rango -1 a 1
    x = np.random.uniform(-1, 1, size=2)
    history = []  # Almacenar el historial de valores para graficar la convergencia
    
    for _ in range(iterations):
        grad = gradient(*x)
        x -= lr * grad

        # Mantener los valores de x en el rango [-1, 1]
        x = np.clip(x, -1, 1)
        
        # Calcular el valor de la función en el punto actual y añadirlo al historial
        history.append(f(*x))

    return x, history

# Parámetros
lr = 0.1  # Learning rate
iterations = 1000

# Ejecutar el descenso del gradiente
optimal_point, convergence_history = gradient_descent(lr, iterations)

# Mostrar el resultado
print("Resultados:")
print("Valor óptimo de X1:", optimal_point[0])
print("Valor óptimo de X2:", optimal_point[1])
print("Valor óptimo de la función:", f(*optimal_point))

# Graficar la convergencia del error
plt.plot(convergence_history)
plt.title("Convergencia del error")
plt.xlabel("Iteración")
plt.ylabel("Valor de la función")
plt.show()
