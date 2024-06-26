# Importar bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Función para cargar el dataset zoo2 o zoo3
def load_zoo_dataset(zoo_file):
    data = pd.read_csv(zoo_file)
    X = data.drop(columns=['animal_name', 'class_type'])
    y = data['class_type']
    return X, y

# Función para preprocesar los datos (escalar características)
def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Función para entrenar y evaluar modelos
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("F1 Score: {:.2f}%".format(f1 * 100))
    print()

# Inicializar modelos de clasificación
models = [
    ("Regresion Logistica", LogisticRegression()),
    ("K-Vecinos Cercanos", KNeighborsClassifier()),
    ("Maquinas de Vectores de Soporte (SVM)", SVC(kernel='linear')),
    ("Naive Bayes", GaussianNB()),
    ("Red Neuronal", MLPClassifier(hidden_layer_sizes=(40, 50, 40), max_iter=1000))
]

# Cargar y preparar el dataset zoo
zoo_file = "datasets/zoo.csv"
X_zoo, y_zoo = load_zoo_dataset(zoo_file)
X_train_zoo, X_test_zoo, y_train_zoo, y_test_zoo = train_test_split(X_zoo, y_zoo, test_size=0.2, random_state=42)
X_train_zoo_scaled, X_test_zoo_scaled = preprocess_data(X_train_zoo, X_test_zoo)

# Entrenar y evaluar modelos para el dataset zoo
print("=== Resultados para Zoo Dataset: ===\n")
for model_name, model in models:
    print("Modelo:", model_name)
    train_and_evaluate_model(model, X_train_zoo_scaled, X_test_zoo_scaled, y_train_zoo, y_test_zoo)

# Cargar y preparar el dataset zoo2
zoo2_file = "datasets/zoo2.csv"
X_zoo2, y_zoo2 = load_zoo_dataset(zoo2_file)
X_train_zoo2, X_test_zoo2, y_train_zoo2, y_test_zoo2 = train_test_split(X_zoo2, y_zoo2, test_size=0.2, random_state=42)
X_train_zoo2_scaled, X_test_zoo2_scaled = preprocess_data(X_train_zoo2, X_test_zoo2)

# Entrenar y evaluar modelos para el dataset zoo2
print("\n=== Resultados para Zoo2 Dataset: ===\n")
for model_name, model in models:
    print("Modelo:", model_name)
    train_and_evaluate_model(model, X_train_zoo2_scaled, X_test_zoo2_scaled, y_train_zoo2, y_test_zoo2)

# Cargar y preparar el dataset zoo3
zoo3_file = "datasets/zoo3.csv"
X_zoo3, y_zoo3 = load_zoo_dataset(zoo3_file)
X_train_zoo3, X_test_zoo3, y_train_zoo3, y_test_zoo3 = train_test_split(X_zoo3, y_zoo3, test_size=0.2, random_state=42)
X_train_zoo3_scaled, X_test_zoo3_scaled = preprocess_data(X_train_zoo3, X_test_zoo3)

# Entrenar y evaluar modelos para el dataset zoo3
print("\n=== Resultados para Zoo3 Dataset: ===\n")
for model_name, model in models:
    print("Modelo:", model_name)
    train_and_evaluate_model(model, X_train_zoo3_scaled, X_test_zoo3_scaled, y_train_zoo3, y_test_zoo3)

# Concatenar los datasets verticalmente
X_combined = pd.concat([X_zoo, X_zoo2, X_zoo3], axis=0)
y_combined = pd.concat([y_zoo, y_zoo2, y_zoo3], axis=0)

# Dividir los datos combinados en conjuntos de entrenamiento y prueba
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Preprocesar los datos combinados
X_train_combined_scaled, X_test_combined_scaled = preprocess_data(X_train_combined, X_test_combined)

# Entrenar y evaluar modelos para el dataset combinado
print("=== Resultados para Dataset Combinado ===\n")
for model_name, model in models:
    print("Modelo:", model_name)
    train_and_evaluate_model(model, X_train_combined_scaled, X_test_combined_scaled, y_train_combined, y_test_combined)
