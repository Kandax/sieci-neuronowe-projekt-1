import numpy as np
from network import NeuralNetwork1Hidden
from plots import (
    generate_plots
)


import csv
import numpy as np


def get_cancer_data(filepath="data.csv"):
    X_list = []
    Y_list = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for parts in reader:
            if len(parts) < 32:
                continue

            diag = parts[1].strip().strip('"')
            # Pomijamy wiersz nagłówka
            if diag == "diagnosis":
                continue

            # M = 1.0 (złośliwy), B = 0.0 (łagodny)
            label = 1.0 if diag == "M" else 0.0

            # Pobieramy 30 cech numerycznych (kolumny 2..31)
            features = [float(val.strip().strip('"')) for val in parts[2:32]]

            X_list.append(features)
            Y_list.append(label)

    X_raw = np.array(X_list, dtype=float)  # kształt: (N, 30)

    # Normalizacja Min-Max każdej kolumny do zakresu [0, 1]
    min_vals = X_raw.min(axis=0, keepdims=True)
    max_vals = X_raw.max(axis=0, keepdims=True)
    denom = np.where((max_vals - min_vals) == 0, 1.0, max_vals - min_vals)
    X_norm = (X_raw - min_vals) / denom

    # Transpozycja do formatu wymaganego przez sieć: X -> (30, N), Y -> (1, N)
    X = X_norm.T
    Y = np.array([Y_list], dtype=float)

    return X, Y


if __name__ == "__main__":
    X_all, Y_all = get_cancer_data()

    # 1. Losowy podział na zbiór treningowy (80%) i testowy (20%)
    SEED = 42
    np.random.seed(SEED)
    m_total = X_all.shape[1]
    perm = np.random.permutation(m_total)
    split = int(0.8 * m_total)

    X_train, Y_train = X_all[:, perm[:split]], Y_all[:, perm[:split]]
    X_test, Y_test = X_all[:, perm[split:]], Y_all[:, perm[split:]]

    # Hiperparametry do eksperymentów w sprawozdaniu
    INPUT_SIZE = 30
    HIDDEN_SIZE = 12
    LEARNING_RATE = 0.5
    MAX_EPOCHS = 5000
    TARGET_ERROR = 1e-4

    nn = NeuralNetwork1Hidden(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=1,
        seed=SEED,
    )

    history, final_epoch = nn.train(
        X_train,
        Y_train,
        learning_rate=LEARNING_RATE,
        epochs=MAX_EPOCHS,
        batch_size=32,
        target_error=TARGET_ERROR,
        momentum=0.9,
        adaptive_lr=True,
        lr_inc=1.05,
        lr_dec=0.7,
        max_error_ratio=1.04,
    )

    # 2. Podsumowanie wyników dla zbioru treningowego i testowego
    train_out, _ = nn.forward(X_train)
    train_acc = np.mean((train_out >= 0.5) == Y_train) * 100

    test_out, _ = nn.forward(X_test)
    test_mse = float(np.mean((Y_test - test_out) ** 2))
    test_errors = int(np.sum((test_out >= 0.5) != Y_test))
    test_acc = np.mean((test_out >= 0.5) == Y_test) * 100

    print(f"Zakończono na epoce: {final_epoch} / {MAX_EPOCHS}")
    print(f"Trening -> MSE: {history['mse_out_total'][-1]:.6f} | Accuracy: {train_acc:.2f}%")
    print(f"Test    -> MSE: {test_mse:.6f} | Accuracy: {test_acc:.2f}% (błędy: {test_errors} / {X_test.shape[1]})\n")

    print("Wejście | Oczekiwane | Predykcja sieci | Po progowaniu (0-1)")
    for x_col, y_true, y_pred in zip(X_test.T, Y_test.T, test_out.T):
        pred_bin = int(y_pred[0] >= 0.5)
        x_str = ",".join(f"{v:.2f}" for v in x_col)
        print(
            f" {x_str}   |     {int(y_true[0])}      |"
            f"     {y_pred[0]:.4f}      |          {pred_bin}"
        )

    generate_plots(history)
