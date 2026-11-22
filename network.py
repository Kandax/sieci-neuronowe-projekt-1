import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1.0 - a)

class NeuralNetwork1Hidden:
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        np.random.seed(seed)
        self.W_1 = np.random.uniform(-1.0, 1.0, size=(hidden_size, input_size))
        self.b_1 = np.zeros((hidden_size, 1))
        self.W_2 = np.random.uniform(-1.0, 1.0, size=(output_size, hidden_size))
        self.b_2 = np.zeros((output_size, 1))


    def forward(self, X):
        Z1 = self.W_1 @ X + self.b_1
        A1 = sigmoid(Z1)

        Z_2 = self.W_2 @ A1 + self.b_2
        A_2 = sigmoid(Z_2)

        cache = {"Z1": Z1, "A1": A1, "Z_2": Z_2, "A_2": A_2}
        return A_2, cache

    
    def backward(self, X, Y, cache):
        A1 = cache["A1"]
        A_2 = cache["A_2"]

        # Błąd i delta warstwy wyjściowej
        E_2 = Y - A_2
        delta_2 = 2 * E_2 * sigmoid_derivative(A_2)

        # Błąd cofnięty i delta warstwy ukrytej
        E1 = self.W_2.T @ delta_2
        delta1 = E1 * sigmoid_derivative(A1)

        # Gradienty wag i biasów (przesunięć)
        grads = {
            "dW_2": delta_2 @ A1.T,
            "db_2": np.sum(delta_2, axis=1, keepdims=True),
            "dW_1": delta1 @ X.T,
            "db_1": np.sum(delta1, axis=1, keepdims=True),
        }
        errors = {"E1": E1, "E_2": E_2}

        return grads, errors


    def update(self, grads, learning_rate):
        self.W_2 += learning_rate * grads["dW_2"]
        self.b_2 += learning_rate * grads["db_2"]
        self.W_1 += learning_rate * grads["dW_1"]
        self.b_1 += learning_rate * grads["db_1"]

    def train(self, X, Y, learning_rate, epochs, target_error=1e-4):
        history = {
            "mse_out_total": [],
            "mse_out_samples": [],
            "mse_hid_total": [],
            "mse_hid_samples": [],
            "class_errors": [],
            "w1": [],
            "w2": [],
        }

        stopped_epoch = epochs

        for epoch in range(1, epochs + 1):
            A_2, cache = self.forward(X)
            grads, errors = self.backward(X, Y, cache)

            # Metryki warstwy wyjściowej
            sq_err_out = errors["E_2"] ** 2
            mse_out_total = float(np.mean(sq_err_out))
            mse_out_samples = sq_err_out.flatten()

            # Metryki warstwy ukrytej
            sq_err_hid = errors["E1"] ** 2
            mse_hid_total = float(np.mean(sq_err_hid))
            mse_hid_samples = np.mean(sq_err_hid, axis=0)

            # Błąd klasyfikacji 0-1 (próg 0.5)
            predictions = (A_2 >= 0.5).astype(float)
            class_error = int(np.sum(predictions != Y))

            # Zapis historii z bieżącej epoki
            history["mse_out_total"].append(mse_out_total)
            history["mse_out_samples"].append(mse_out_samples)
            history["mse_hid_total"].append(mse_hid_total)
            history["mse_hid_samples"].append(mse_hid_samples)
            history["class_errors"].append(class_error)
            history["w1"].append(self.W_1.flatten().copy())
            history["w2"].append(self.W_2.flatten().copy())

            # Warunek stopu - przerywamy przed aktualizacją, jeśli błąd jest minimalny
            if mse_out_total <= target_error:
                stopped_epoch = epoch
                break

            self.update(grads, learning_rate)

        for key in history:
            history[key] = np.array(history[key])

        return history, stopped_epoch