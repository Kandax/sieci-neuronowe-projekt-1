import numpy as np
from network import NeuralNetwork1Hidden
from plots import (
    generate_plots
)


def get_cancer_data():
    # Dane z pliku CSV (bez nagłówka)!!! zaimplementować 
    # M = 1 (szkodliwy rak), B = 0 (łagodny rak)
    # przeskalować dane wejściowe do zakresu [0, 1] (min-max normalization)
    return 1, 1


if __name__ == "__main__":
    X, Y = get_cancer_data()

    # Hiperparametry do eksperymentów w sprawozdaniu
    INPUT_SIZE = 30
    HIDDEN_SIZE = 2  # fuck around with size for better results
    LEARNING_RATE = 0.5
    MAX_EPOCHS = 10000
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
        print(
            f" {int(x_col[0])}, {int(x_col[1])}   |     {int(y_true[0])}      |"
            f"     {y_pred[0]:.4f}      |          {pred_bin}"
        )

    generate_plots(history)
    