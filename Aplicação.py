import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#PROJETO 3

# Carregue seus dados do 'test.csv'
data = 'test.csv'
df = pd.read_csv(data)
df['1.7'] = df['1.7'].replace({1: 0, 2: 1})

# Defina a função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Defina a derivada da função sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Defina a função de treinamento para a MLP com duas camadas ocultas
def initialize_weights(input_size, hidden_units1, hidden_units2, output_size):
    limit_xavier_hidden1 = np.sqrt(2 / (input_size + hidden_units1))
    weights_hidden1 = np.random.uniform(-limit_xavier_hidden1, limit_xavier_hidden1, (input_size, hidden_units1))

    limit_xavier_hidden2 = np.sqrt(2 / (hidden_units1 + hidden_units2))
    weights_hidden2 = np.random.uniform(-limit_xavier_hidden2, limit_xavier_hidden2, (hidden_units1, hidden_units2))

    limit_xavier_output = np.sqrt(2 / (hidden_units2 + output_size))
    weights_output = np.random.uniform(-limit_xavier_output, limit_xavier_output, (hidden_units2, output_size))

    return weights_hidden1, weights_hidden2, weights_output

# Parte II: Implemente a propagação para frente e calcule o custo
def forward_propagation(X, weights_hidden1, weights_hidden2, weights_output):
    hidden1 = sigmoid(np.dot(X, weights_hidden1))
    hidden2 = sigmoid(np.dot(hidden1, weights_hidden2))
    output = sigmoid(np.dot(hidden2, weights_output))
    return hidden1, hidden2, output

def calculate_cost(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    m = len(y_true)
    cost = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return cost

# Parte III: Calcule o gradiente da função de custo
def calculate_gradients(X, y, hidden1, hidden2, output, weights_hidden1, weights_hidden2, weights_output):
    delta_output = output - y
    delta_hidden2 = delta_output.dot(weights_output.T) * sigmoid_derivative(hidden2)
    delta_hidden1 = delta_hidden2.dot(weights_hidden2.T) * sigmoid_derivative(hidden1)

    grad_output = hidden2.T.dot(delta_output)
    grad_hidden2 = hidden1.T.dot(delta_hidden2)
    grad_hidden1 = X.T.dot(delta_hidden1)

    return grad_hidden1, grad_hidden2, grad_output

# Parte IV: Implemente a retropropagação. (Use a regra de atualização para o gradiente descendente)
def backpropagation(weights_hidden1, weights_hidden2, weights_output, grad_hidden1, grad_hidden2, grad_output, learning_rate):
    weights_hidden1 -= learning_rate * grad_hidden1
    weights_hidden2 -= learning_rate * grad_hidden2
    weights_output -= learning_rate * grad_output

    return weights_hidden1, weights_hidden2, weights_output

# Parte completa do treinamento
def train_neural_network(X, y, hidden_units1, hidden_units2, output_size, learning_rate, epochs):
    input_size = X.shape[1]

    weights_hidden1, weights_hidden2, weights_output = initialize_weights(input_size, hidden_units1, hidden_units2, output_size)

    for epoch in range(epochs):
        hidden1, hidden2, output = forward_propagation(X, weights_hidden1, weights_hidden2, weights_output)
        cost = calculate_cost(y, output)
        grad_hidden1, grad_hidden2, grad_output = calculate_gradients(X, y, hidden1, hidden2, output, weights_hidden1, weights_hidden2, weights_output)
        weights_hidden1, weights_hidden2, weights_output = backpropagation(weights_hidden1, weights_hidden2, weights_output, grad_hidden1, grad_hidden2, grad_output, learning_rate)

        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Cost: {cost}")

    return weights_hidden1, weights_hidden2, weights_output

# Dividir os dados em conjuntos de treinamento e teste
y = df['1.7'].values.reshape(-1, 1)
X = df.drop(columns=['1.7'])
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
scaler.fit(X_treinamento)
X_treinamento = scaler.transform(X_treinamento)
X_teste = scaler.transform(X_teste)

# Treinar a rede neural
hidden_units1 = 50
hidden_units2 = 50
output_size = 1
learning_rate = 0.01
epochs = 100

trained_weights_hidden1, trained_weights_hidden2, trained_weights_output = train_neural_network(X_treinamento, y_treinamento, hidden_units1, hidden_units2, output_size, learning_rate, epochs)

# Avaliar a rede neural no conjunto de teste
_, _, y_predito = forward_propagation(X_teste, trained_weights_hidden1, trained_weights_hidden2, trained_weights_output)
y_predito = (y_predito > 0.5).astype(int)

# Avaliar a precisão do modelo
acuracia = accuracy_score(y_teste, y_predito)
print("Acurácia da Rede Neural:", acuracia)
custo_teste = calculate_cost(y_teste, y_predito)
media_custo_teste = np.mean(custo_teste)
print(f'Média da Função Custo no Conjunto de Teste: {media_custo_teste}')