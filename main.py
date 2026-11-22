import numpy as np
from network import NeuralNetwork1Hidden
from plots import (
    plot_classification_error,
    plot_mse,
    plot_learning_rate,
    plot_weights,
)


def get_xor_data():
    # Przykłady jako kolumny: X ma kształt (2, 4), Y ma kształt (1, 4)
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=float)
    Y = np.array([[0, 1, 1, 0]], dtype=float)
    return X, Y


if __name__ == "__main__":
    X, Y = get_xor_data()

    # Hiperparametry do eksperymentów w sprawozdaniu
    HIDDEN_SIZE = 2
    LEARNING_RATE = 0.5
    MAX_EPOCHS = 10000
    TARGET_ERROR = 0.001
    SEED = 42

    nn = NeuralNetwork1Hidden(
        input_size=2,
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

    # Generowanie kompletu wykresów do sprawozdania
    plot_mse(
        history["mse_out_total"],
        history["mse_out_samples"],
        layer_name="warstwa wyjściowa",
        filename="mse_wyjsciowa.png",
    )
    plot_mse(
        history["mse_hid_total"],
        history["mse_hid_samples"],
        layer_name="warstwa ukryta",
        filename="mse_ukryta.png",
    )
    plot_classification_error(
        history["class_errors"],
        filename="blad_klasyfikacji.png",
    )
    plot_weights(
        history["w1"],
        layer_name="warstwa ukryta (W1)",
        filename="wagi_ukryta.png",
    )
    plot_weights(
        history["w2"],
        layer_name="warstwa wyjściowa (W2)",
        filename="wagi_wyjsciowa.png",
    )
    plot_learning_rate(
        history["lr"],
        filename="learning_rate.png",
    )