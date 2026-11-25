import numpy as np


def sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1.0 - a)

class NeuralNetwork1Hidden:
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        np.random.seed(seed)

        # wagi i biasy inicjalizowane losowo w zakresie [-1, 1]
        self.W_1 = np.random.uniform(-1.0, 1.0, size=(hidden_size, input_size))
        self.b_1 = np.zeros((hidden_size, 1))
        self.W_2 = np.random.uniform(-1.0, 1.0, size=(output_size, hidden_size))
        self.b_2 = np.zeros((output_size, 1))

        # macierze prędkości (momentum) zainicjalizowane zerami
        self.vW_1 = np.zeros_like(self.W_1)
        self.vb_1 = np.zeros_like(self.b_1)
        self.vW_2 = np.zeros_like(self.W_2)
        self.vb_2 = np.zeros_like(self.b_2)


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


    def update(self, grads, learning_rate, momentum=0.9):
        # obliczenie nowej prędkości
        self.vW_2 = momentum * self.vW_2 + learning_rate * grads["dW_2"]
        self.vb_2 = momentum * self.vb_2 + learning_rate * grads["db_2"]
        self.vW_1 = momentum * self.vW_1 + learning_rate * grads["dW_1"]
        self.vb_1 = momentum * self.vb_1 + learning_rate * grads["db_1"]

        # aktualizacja wag i biasów o wyliczoną prędkość
        self.W_2 += self.vW_2
        self.b_2 += self.vb_2
        self.W_1 += self.vW_1
        self.b_1 += self.vb_1

    def train(
        self,
        X,
        Y,
        learning_rate,
        epochs,
        batch_size=None,  # None oznacza cały zbiór (Full Batch), np. 4 dla XOR
        target_error=1e-4,
        momentum=0.9,
        adaptive_lr=True,
        lr_inc=1.05,
        lr_dec=0.7,
        max_error_ratio=1.04,
    ):
        m = X.shape[1]
        if batch_size is None or batch_size > m:
            batch_size = m

        history = {
            "mse_out_total": [],
            "mse_out_samples": [],
            "mse_hid_total": [],
            "mse_hid_samples": [],
            "class_errors": [],
            "w1": [],
            "w2": [],
            "lr": [],
        }

        stopped_epoch = epochs
        prev_mse = None

        for epoch in range(1, epochs + 1):
            # 1. Ewaluacja na całym zbiorze X przed krokiem (do metryk, wykresów i warunku stopu)
            A_2, cache = self.forward(X)
            _, errors = self.backward(X, Y, cache)

            sq_err_out = errors["E_2"] ** 2
            mse_out_total = float(np.mean(sq_err_out))
            mse_out_samples = sq_err_out.flatten()

            sq_err_hid = errors["E1"] ** 2
            mse_hid_total = float(np.mean(sq_err_hid))
            mse_hid_samples = np.mean(sq_err_hid, axis=0)

            predictions = (A_2 >= 0.5).astype(float)
            class_error = int(np.sum(predictions != Y))

            history["mse_out_total"].append(mse_out_total)
            history["mse_out_samples"].append(mse_out_samples)
            history["mse_hid_total"].append(mse_hid_total)
            history["mse_hid_samples"].append(mse_hid_samples)
            history["class_errors"].append(class_error)
            history["w1"].append(self.W_1.flatten().copy())
            history["w2"].append(self.W_2.flatten().copy())
            history["lr"].append(learning_rate)

            if mse_out_total <= target_error:
                stopped_epoch = epoch
                break

            # 2. Adaptacja learning_rate na podstawie pełnego błędu epoki
            if adaptive_lr and prev_mse is not None:
                if mse_out_total > max_error_ratio * prev_mse:
                    self.W_1, self.b_1 = prev_params["W_1"], prev_params["b_1"]
                    self.W_2, self.b_2 = prev_params["W_2"], prev_params["b_2"]
                    self.vW_1.fill(0.0)
                    self.vb_1.fill(0.0)
                    self.vW_2.fill(0.0)
                    self.vb_2.fill(0.0)
                    learning_rate *= lr_dec
                elif mse_out_total < prev_mse:
                    learning_rate *= lr_inc

            prev_mse = mse_out_total
            prev_params = {
                "W_1": self.W_1.copy(),
                "b_1": self.b_1.copy(),
                "W_2": self.W_2.copy(),
                "b_2": self.b_2.copy(),
            }

            # 3. Właściwe uczenie w paczkach (mini-batch) z losowym tasowaniem
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i : i + batch_size]
                Y_batch = Y_shuffled[:, i : i + batch_size]

                _, batch_cache = self.forward(X_batch)
                grads, _ = self.backward(X_batch, Y_batch, batch_cache)

                # Uśrednienie gradientu przez aktualny rozmiar paczki (lub przez m dla spójnej skali)
                scale = X_batch.shape[1] / m
                self.update(grads, learning_rate * scale, momentum=momentum)

        for key in history:
            history[key] = np.array(history[key])

        return history, stopped_epoch