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
    X, Y = get_cancer_data()

    # Hiperparametry do eksperymentów w sprawozdaniu
    INPUT_SIZE = 30
    HIDDEN_SIZE = 30  # fuck around with size for better results
    LEARNING_RATE = 0.5
    MAX_EPOCHS = 100000
    TARGET_ERROR = 0.001
    SEED = 42

    nn = NeuralNetwork1Hidden(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=1,
        seed=SEED,
    )

    history, final_epoch = nn.train(
            X,
            Y,
            learning_rate=LEARNING_RATE,
            epochs=MAX_EPOCHS,
            batch_size=None,  # None oznacza cały zbiór (Full Batch), np. 4 dla XOR
            target_error=1e-4,
            momentum=0.9,
            adaptive_lr=True,
            lr_inc=1.05,
            lr_dec=0.7,
            max_error_ratio=1.04,
        )

    # Podsumowanie wyników w konsoli
    final_output, _ = nn.forward(X)
    print(f"Zakończono na epoce: {final_epoch} / {MAX_EPOCHS}")
    print(f"Końcowe MSE warstwy wyjściowej: {history['mse_out_total'][-1]:.6f}")
    print(f"Końcowy błąd klasyfikacji: {history['class_errors'][-1]} / 4\n")

    print("Wejście | Oczekiwane | Predykcja sieci | Po progowaniu (0-1)")
    for x_col, y_true, y_pred in zip(X.T, Y.T, final_output.T):
        pred_bin = int(y_pred[0] >= 0.5)
        x_str = ",".join(f"{v:.2f}" for v in x_col)
        print(
            f" {x_str}   |     {int(y_true[0])}      |"
            f"     {y_pred[0]:.4f}      |          {pred_bin}"
        )

    generate_plots(history)
    