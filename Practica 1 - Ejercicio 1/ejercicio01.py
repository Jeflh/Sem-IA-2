import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.07, max_epochs=100):
        limit = np.sqrt(6 / (input_size + 1))  # Limit es la desviación estándar deseada
        self.weights = np.random.uniform(-limit, limit, size=(input_size + 1))
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def train(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Inserting bias term
        for epoch in range(self.max_epochs):
            for i in range(X.shape[0]):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]

    def predict(self, x):
        return 1 if np.dot(self.weights, x) > 0 else -1

def plot_data_and_separator(X_train, y_train, X_test, y_test, weights):
    plt.figure()
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test )
    plt.xlabel('X1')
    plt.ylabel('X2')

    # Plotting decision boundary
    slope = -weights[1] / weights[2]
    intercept = -weights[0] / weights[2]
    x_vals = np.array(plt.gca().get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--r', label='Límite de decisión')

    plt.legend()
    plt.show()

def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

if __name__ == "__main__":
    # Cargar datos de entrenamiento y prueba
    X_train, y_train = load_data('XOR_trn.csv')
    X_test, y_test = load_data('XOR_tst.csv')

    # Entrenar perceptrón
    perceptron = Perceptron(input_size=X_train.shape[1])
    perceptron.train(X_train, y_train)

    # Probar perceptrón en datos de prueba y calcular precisión
    correct = 0
    for i in range(X_test.shape[0]):
        prediction = perceptron.predict(np.insert(X_test[i], 0, 1))
        if prediction == y_test[i]:
            correct += 1

    accuracy = correct / len(y_test)
    print(f'Precisión en datos de prueba XOR: {accuracy * 100:.2f}%')

    # Visualizar los datos y la recta que separa las clases
    plot_data_and_separator(X_train, y_train, X_test, y_test, perceptron.weights)


    # Ahora con OR
    X_train, y_train = load_data('OR_trn.csv')
    X_test, y_test = load_data('OR_tst.csv')

    # Entrenar perceptrón
    perceptron = Perceptron(input_size=X_train.shape[1])
    perceptron.train(X_train, y_train)
    
    # Probar perceptrón en datos de prueba y calcular precisión
    correct = 0
    for i in range(X_test.shape[0]):
        prediction = perceptron.predict(np.insert(X_test[i], 0, 1))
        if prediction == y_test[i]:
            correct += 1
    
    accuracy = correct / len(y_test)
    print(f'Precisión en datos de prueba OR: {accuracy * 100:.2f}%')

    # Visualizar los datos y la recta que separa las clases
    plot_data_and_separator(X_train, y_train, X_test, y_test, perceptron.weights)
