import pandas as pd
import numpy as np

# Perceptrón Simple
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.n_iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# Función para cargar datos desde un archivo CSV sin nombres de columna
def load_data(file_path):
    return pd.read_csv(file_path, header=None)

# Función para generar particiones de entrenamiento y prueba
def generate_partitions(data, train_size, random_state=None):
    np.random.seed(random_state)
    np.random.shuffle(data)
    train_len = int(len(data) * train_size)
    return data[:train_len], data[train_len:]

# Función para entrenar el perceptrón simple
def train_perceptron(X_train, X_test):
    perceptron = Perceptron()
    perceptron.fit(X_train[:, :-1], X_train[:, -1])
    predictions = perceptron.predict(X_test[:, :-1])
    accuracy = np.mean(predictions == X_test[:, -1])
    return accuracy

# Archivo: spheres1d10.csv
# Cargar datos
data = load_data("spheres1d10.csv")
print(f"Archivo: spheres1d10.csv\n")
# Generar particiones y entrenar el perceptrón simple
for i in range(5):
    X_train, X_test = generate_partitions(data.to_numpy(), train_size=0.8, random_state=i)
    accuracy = train_perceptron(X_train, X_test) * 100
    print(f"Partición {i+1}: Precisión = {accuracy:.1f}%")

# Archivo: spheres2d10.csv
data = load_data("spheres2d10.csv")
print(f"\nArchivo: spheres2d10.csv\n")
for i in range(10):
    X_train, X_test = generate_partitions(data.to_numpy(), train_size=0.8, random_state=i)
    accuracy = train_perceptron(X_train, X_test) * 100
    print(f"Partición {i+1}: Precisión = {accuracy:.1f}%")

# Archivo: spheres2d50.csv
data = load_data("spheres2d50.csv")
print(f"\nArchivo: spheres2d50.csv\n")
for i in range(10):
    X_train, X_test = generate_partitions(data.to_numpy(), train_size=0.8, random_state=i)
    accuracy = train_perceptron(X_train, X_test) * 100
    print(f"Partición {i+1}: Precisión = {accuracy:.1f}%")

# Archivo: spheres2d70.csv
data = load_data("spheres2d70.csv")
print(f"\nArchivo: spheres2d70.csv\n")
for i in range(10):
    X_train, X_test = generate_partitions(data.to_numpy(), train_size=0.8, random_state=i)
    accuracy = train_perceptron(X_train, X_test) * 100
    print(f"Partición {i+1}: Precisión = {accuracy:.1f}%")
