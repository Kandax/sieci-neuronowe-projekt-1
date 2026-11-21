import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1.0 - a)


def init_parameters(input_size, hidden_size, output_size, seed=42):
    np.random.seed(seed)
    return {
        "W1": np.random.uniform(-1.0, 1.0, size=(input_size, hidden_size)),
        "b1": np.zeros((1, hidden_size)),
        "W2": np.random.uniform(-1.0, 1.0, size=(hidden_size, output_size)),
        "b2": np.zeros((1, output_size)),
    }


def forward_pass(X, params):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = sigmoid(Z1)

    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = sigmoid(Z2)

    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}


def backward_pass(X, Y, params, cache):
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Błąd i delta warstwy wyjściowej
    E2 = Y - A2
    delta2 = E2 * sigmoid_derivative(A2)

    # Błąd cofnięty i delta warstwy ukrytej
    E1 = delta2 @ params["W2"].T
    delta1 = E1 * sigmoid_derivative(A1)

    # Gradienty wag i biasów (przesunięć)
    grads = {
        "dW2": A1.T @ delta2,
        "db2": np.sum(delta2, axis=0, keepdims=True),
        "dW1": X.T @ delta1,
        "db1": np.sum(delta1, axis=0, keepdims=True),
    }
    errors = {"E1": E1, "E2": E2}

    return grads, errors


def update_parameters(params, grads, learning_rate):
    params["W2"] += learning_rate * grads["dW2"]
    params["b2"] += learning_rate * grads["db2"]
    params["W1"] += learning_rate * grads["dW1"]
    params["b1"] += learning_rate * grads["db1"]